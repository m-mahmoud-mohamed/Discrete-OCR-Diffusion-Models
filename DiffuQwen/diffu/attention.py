"""
Annealed Attention Mask Builder for DiffuQwen-VL.

Implements the attention mask annealing strategy from DiffuLLaMA:
- Gradual transition from causal to bidirectional attention
- Multimodal attention rules for vision-language models
- Compatible with Qwen2.5-VL's M-RoPE

Attention Mask Rules:
| Interaction    | Mask Type     | Reason                              |
|----------------|---------------|-------------------------------------|
| Image→Image    | Full (1)      | Visual tokens always see each other |
| Image→Text     | Full (1)      | Visual tokens can see text          |
| Text→Image     | Full (1)      | Text MUST see full image for OCR    |
| Text→Text      | ANNEALING     | Causal → Bidirectional over 10k steps|
"""

import torch
from typing import Optional, Tuple


def build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build a causal (lower triangular) attention mask.
    
    Args:
        seq_len: Sequence length
        device: Target device
        dtype: Target dtype
    
    Returns:
        Causal mask of shape (seq_len, seq_len), 0 for attend, -inf for mask
    """
    # Create lower triangular matrix (1s below diagonal, 0s above)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    # Convert to attention mask format: 0 = attend, -inf = mask
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.to(dtype)


def build_full_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build a full (bidirectional) attention mask.
    
    Args:
        seq_len: Sequence length
        device: Target device
        dtype: Target dtype
    
    Returns:
        Full mask of shape (seq_len, seq_len), all zeros (full attention)
    """
    return torch.zeros(seq_len, seq_len, device=device, dtype=dtype)


def build_annealed_attention_mask(
    seq_len: int,
    anneal_progress: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build an annealed attention mask transitioning from causal to bidirectional.
    
    At anneal_progress=0: Pure causal mask
    At anneal_progress=1: Pure bidirectional mask
    Between: Probabilistically unmask upper triangle positions
    
    Args:
        seq_len: Sequence length
        anneal_progress: Progress in [0, 1], where 0=causal, 1=bidirectional
        device: Target device
        dtype: Target dtype
    
    Returns:
        Annealed mask of shape (seq_len, seq_len)
    """
    assert 0.0 <= anneal_progress <= 1.0, f"anneal_progress must be in [0, 1], got {anneal_progress}"
    
    if anneal_progress == 0.0:
        return build_causal_mask(seq_len, device, dtype)
    elif anneal_progress >= 1.0:
        return build_full_mask(seq_len, device, dtype)
    
    # Start with causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    
    # Probabilistically unmask upper triangle based on progress
    # Sample random values for upper triangle positions
    upper_triangle_mask = mask == 1
    rand_vals = torch.rand(seq_len, seq_len, device=device)
    
    # Unmask positions where random value < anneal_progress
    should_unmask = (rand_vals < anneal_progress) & upper_triangle_mask
    mask[should_unmask] = 0
    
    # Convert to attention format
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.to(dtype)


def build_multimodal_attention_mask(
    batch_size: int,
    seq_len: int,
    image_token_count: int,
    anneal_progress: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    prompt_len: int = 0,
) -> torch.Tensor:
    """
    Build multimodal attention mask with annealing for vision-language models.
    
    Mask structure:
    - Image→Image: Full attention (always)
    - Image→Text: Full attention (always)
    - Text→Image: Full attention (always, critical for OCR)
    - Text→Text: Annealed (causal→bidirectional)
    
    Args:
        batch_size: Batch size
        seq_len: Total sequence length (image + text tokens)
        image_token_count: Number of image tokens at the start
        anneal_progress: Progress in [0, 1] for text→text annealing
        device: Target device
        dtype: Target dtype
        prompt_len: Length of prompt tokens (after image, before output)
    
    Returns:
        Attention mask of shape (batch_size, 1, seq_len, seq_len)
    """
    # Initialize full attention mask (all zeros = full attention)
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    # Define regions
    # [IMAGE TOKENS | PROMPT TOKENS | OUTPUT TOKENS]
    text_start = image_token_count
    output_start = image_token_count + prompt_len
    
    # Text→Text region (only output tokens participate in annealing)
    # The output region uses annealed attention
    if output_start < seq_len:
        output_len = seq_len - output_start
        
        if anneal_progress < 1.0:
            # Build annealed mask for output→output attention
            output_mask = torch.triu(
                torch.ones(output_len, output_len, device=device), diagonal=1
            )
            
            if anneal_progress > 0.0:
                # Probabilistically unmask
                rand_vals = torch.rand(output_len, output_len, device=device)
                upper_triangle = output_mask == 1
                should_unmask = (rand_vals < anneal_progress) & upper_triangle
                output_mask[should_unmask] = 0
            
            # Convert to attention format
            output_mask = output_mask.masked_fill(output_mask == 1, float("-inf"))
            
            # Place in the output→output region
            mask[output_start:, output_start:] = output_mask.to(dtype)
    
    # Expand for batch and heads: (batch_size, 1, seq_len, seq_len)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    
    return mask


def build_deterministic_annealed_mask(
    seq_len: int,
    anneal_progress: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a deterministic annealed mask (no randomness).
    
    Uses a fixed pattern where positions are unmasked based on their
    relative position in the upper triangle.
    
    Args:
        seq_len: Sequence length
        anneal_progress: Progress in [0, 1]
        device: Target device
        dtype: Target dtype
    
    Returns:
        Deterministic annealed mask of shape (seq_len, seq_len)
    """
    if anneal_progress >= 1.0:
        return build_full_mask(seq_len, device, dtype)
    elif anneal_progress <= 0.0:
        return build_causal_mask(seq_len, device, dtype)
    
    # Create position indices
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Distance from diagonal (0 on diagonal, 1 just above, etc.)
    distance = cols - rows
    
    # Max distance in upper triangle
    max_distance = seq_len - 1
    
    # Unmask positions where normalized distance <= anneal_progress
    # Closer to diagonal gets unmasked first
    normalized_distance = distance.float() / max_distance
    
    # Mask: 1 where should be masked, 0 where should attend
    mask = (distance > 0) & (normalized_distance > anneal_progress)
    mask = mask.float().masked_fill(mask == 1, float("-inf"))
    
    return mask.to(dtype)


class AnnealedAttentionMaskBuilder:
    """
    Builder class for annealed attention masks with state tracking.
    
    Tracks global step and computes anneal progress automatically.
    """
    
    def __init__(
        self,
        anneal_steps: int = 10000,
        deterministic: bool = False,
    ):
        """
        Initialize the mask builder.
        
        Args:
            anneal_steps: Number of steps for causal→bidirectional transition
            deterministic: If True, use deterministic masking pattern
        """
        self.anneal_steps = anneal_steps
        self.deterministic = deterministic
        self._global_step = 0
    
    @property
    def global_step(self) -> int:
        """Current global step."""
        return self._global_step
    
    @global_step.setter
    def global_step(self, value: int):
        """Set global step."""
        self._global_step = max(0, value)
    
    @property
    def anneal_progress(self) -> float:
        """Current annealing progress in [0, 1]."""
        return min(1.0, self._global_step / self.anneal_steps)
    
    @property
    def is_fully_bidirectional(self) -> bool:
        """Whether annealing is complete (full bidirectional)."""
        return self._global_step >= self.anneal_steps
    
    def build_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Build annealed attention mask for current step.
        
        Args:
            seq_len: Sequence length
            device: Target device
            dtype: Target dtype
        
        Returns:
            Attention mask of shape (seq_len, seq_len)
        """
        if self.deterministic:
            return build_deterministic_annealed_mask(
                seq_len, self.anneal_progress, device, dtype
            )
        else:
            return build_annealed_attention_mask(
                seq_len, self.anneal_progress, device, dtype
            )
    
    def build_multimodal_mask(
        self,
        batch_size: int,
        seq_len: int,
        image_token_count: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        prompt_len: int = 0,
    ) -> torch.Tensor:
        """
        Build multimodal attention mask for current step.
        
        Args:
            batch_size: Batch size
            seq_len: Total sequence length
            image_token_count: Number of image tokens
            device: Target device
            dtype: Target dtype
            prompt_len: Length of prompt tokens
        
        Returns:
            Attention mask of shape (batch_size, 1, seq_len, seq_len)
        """
        return build_multimodal_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            image_token_count=image_token_count,
            anneal_progress=self.anneal_progress,
            device=device,
            dtype=dtype,
            prompt_len=prompt_len,
        )
    
    def step(self):
        """Increment global step by 1."""
        self._global_step += 1
    
    def __repr__(self) -> str:
        return (
            f"AnnealedAttentionMaskBuilder(step={self._global_step}/{self.anneal_steps}, "
            f"progress={self.anneal_progress:.4f}, deterministic={self.deterministic})"
        )


def get_attention_mask_for_inference(
    batch_size: int,
    seq_len: int,
    image_token_count: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Get full bidirectional attention mask for inference.
    
    During inference, we always use full bidirectional attention
    (annealing is only for training).
    
    Args:
        batch_size: Batch size
        seq_len: Total sequence length
        image_token_count: Number of image tokens
        device: Target device
        dtype: Target dtype
    
    Returns:
        Full attention mask of shape (batch_size, 1, seq_len, seq_len)
    """
    return build_multimodal_attention_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        image_token_count=image_token_count,
        anneal_progress=1.0,  # Full bidirectional
        device=device,
        dtype=dtype,
        prompt_len=0,
    )
