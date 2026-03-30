import math
import torch
import torch.nn as nn
import re
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from .pooler_projector import PoolerProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    
class Qwen2VLPatchMerger(nn.Module):
    """
    Spatial Patch Merger adapted from Qwen2-VL.

    This module performs 2x2 spatial patch merging through depth-wise concatenation
    followed by MLP projection. It reduces spatial dimensions by 2x while preserving
    local spatial details critical for OCR and document understanding.

    Args:
        config: Model configuration with the following attributes:
            - mm_hidden_size: Vision encoder output dimension (e.g., 1152 for SigLIP)
            - hidden_size: LLM input dimension (e.g., 4096 for Llama-7B)
            - spatial_merge_size: Merging window size (default: 2)
    """

    def __init__(self, config):
        super().__init__()

        # Extract dimensions from config
        context_dim = config.mm_hidden_size  # Vision encoder dimension
        dim = config.hidden_size             # LLM dimension
        spatial_merge_size = getattr(config, 'spatial_merge_size', 2)

        # After merging 2x2 patches, the feature dimension becomes 4x larger
        self.hidden_size = context_dim * (spatial_merge_size ** 2)

        # Layer Normalization before merging
        # Note: Qwen2-VL uses RMSNorm, but LayerNorm is a safe approximation
        self.ln_q = LlamaRMSNorm(context_dim, eps=1e-6)

        # MLP: Project concatenated features to LLM dimension
        # Architecture: Linear(4*context_dim -> 4*context_dim) -> GELU -> Linear(4*context_dim -> dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing 2x2 patch merging.

        Args:
            x: Vision features of shape (batch_size, seq_len, hidden_dim)
               where seq_len = H * W (spatial grid flattened)

        Returns:
            Merged features of shape (batch_size, seq_len/4, llm_dim)

        Implementation follows the official Qwen2-VL approach:
        1. Apply RMSNorm
        2. Reshape sequence to 2D grid (H, W)
        3. Reorganize into 2x2 blocks and concatenate depth-wise
        4. Project through MLP to LLM dimension
        """
        B, L, C = x.shape
        
        # Step 1: Layer Norm
        x = self.ln_q(x)
        
        # Step 2: Infer spatial dimensions
        H = W = int(math.sqrt(L))
        
        # Handle cases where H*W != L (e.g. if CLS token is present)
        # SigLIP usually has no CLS, so H*W should match L.
        # But if H is odd (e.g. 27), we need to handle it.
        
        # Reshape to grid
        x = x.view(B, H, W, C)
        
        # === FIX START: Pad if dimensions are odd ===
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        
        if pad_h > 0 or pad_w > 0:
            # Pad with zeros: (left, right, top, bottom)
            # F.pad expects (last_dim_left, last_dim_right, ... first_dim_top, first_dim_bottom)
            # We want to pad H (top/bottom) and W (left/right).
            # But here x is (B, H, W, C).
            # Permute to (B, C, H, W) for standard padding or use index slicing?
            # Easier: Just cat zeros.
            
            if pad_h > 0:
                # Pad Height (Bottom)
                zeros = torch.zeros(B, pad_h, W, C, device=x.device, dtype=x.dtype)
                x = torch.cat([x, zeros], dim=1)
                H += pad_h
                
            if pad_w > 0:
                # Pad Width (Right)
                zeros = torch.zeros(B, H, pad_w, C, device=x.device, dtype=x.dtype)
                x = torch.cat([x, zeros], dim=2)
                W += pad_w
        # === FIX END ===

        # Step 3: Perform 2x2 Patch Merging
        # (B, H, W, C) -> (B, H/2, 2, W/2, 2, C)
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H // 2, W // 2, self.hidden_size)
        x = x.view(B, -1, self.hidden_size)
        
        # Step 4: MLP
        x = self.mlp(x)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if not hasattr(config,'hidden_size'):
        config.hidden_size = config.d_model
        
    if projector_type == 'qwen2_vl_merger':
        print("\n" + "="*50)
        print("✅ SUCCESS: Initializing Qwen2-VL Patch Merger!")
        print(f"   Context Dim: {config.mm_hidden_size}")
        print(f"   Hidden Dim:  {config.hidden_size}")
        print("="*50 + "\n")
        return Qwen2VLPatchMerger(config)
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"],pooler_ratio=config.mm_pooler_ratio)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
