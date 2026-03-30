"""
Absorbing State Diffusion Schedule for DiffuQwen-VL.

Implements the linear absorbing state schedule from DiffuLLaMA:
- αₜ = 1 - t (noise level increases linearly with t)
- mask_ratio = 1 - αₜ = t
- At t=0: no masking (clean data)
- At t=1: full masking (all [MASK] tokens)

The model learns to predict clean tokens from partially masked sequences,
inferring the timestep implicitly from the ratio of [MASK] tokens.
"""

import torch
from typing import Tuple, Optional


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    min_t: float = 0.0,
    max_t: float = 1.0,
) -> torch.Tensor:
    """
    Sample uniform timesteps for diffusion training.
    
    Args:
        batch_size: Number of timesteps to sample
        device: Device to place tensor on
        min_t: Minimum timestep (default 0.0)
        max_t: Maximum timestep (default 1.0)
    
    Returns:
        Tensor of shape (batch_size,) with timesteps in [min_t, max_t]
    """
    t = torch.rand(batch_size, device=device) * (max_t - min_t) + min_t
    return t


def get_alpha_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    Compute αₜ for the absorbing state schedule.
    
    Linear schedule: αₜ = 1 - t
    - αₜ = 1 at t=0 (no noise)
    - αₜ = 0 at t=1 (full noise/masking)
    
    Args:
        t: Timesteps tensor of shape (batch_size,) in [0, 1]
    
    Returns:
        Alpha values of shape (batch_size,)
    """
    assert torch.all((t >= 0) & (t <= 1)), f"Timesteps must be in [0, 1], got min={t.min()}, max={t.max()}"
    return 1.0 - t


def get_mask_ratio(t: torch.Tensor) -> torch.Tensor:
    """
    Compute mask ratio from timestep.
    
    mask_ratio = 1 - αₜ = t
    
    Args:
        t: Timesteps tensor of shape (batch_size,) in [0, 1]
    
    Returns:
        Mask ratios of shape (batch_size,)
    """
    return 1.0 - get_alpha_schedule(t)


def apply_absorbing_noise(
    input_ids: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    text_region_mask: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply absorbing state noise to text tokens.
    
    For each text token position, replace with [MASK] with probability (1 - αₜ) = t.
    Image tokens and prompt tokens are NEVER masked.
    
    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        t: Timesteps of shape (batch_size,) in [0, 1]
        mask_token_id: Token ID for [MASK]
        text_region_mask: Boolean mask of shape (batch_size, seq_len), 
                         True for positions that CAN be masked (text output region)
        generator: Optional random generator for reproducibility
    
    Returns:
        x_t: Noised token IDs of shape (batch_size, seq_len)
        noise_mask: Boolean tensor indicating which positions were masked
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Get mask ratio for each sample: shape (batch_size,)
    mask_ratio = get_mask_ratio(t)
    
    # Sample random values for masking decision: shape (batch_size, seq_len)
    rand_vals = torch.rand(batch_size, seq_len, device=device, generator=generator)
    
    # Determine which positions to mask:
    # - Random value < mask_ratio (per-sample threshold)
    # - Position is in text region (can be masked)
    should_mask = (rand_vals < mask_ratio.unsqueeze(1)) & text_region_mask
    
    # Apply masking
    x_t = input_ids.clone()
    x_t[should_mask] = mask_token_id
    
    return x_t, should_mask


def get_inference_timesteps(
    num_steps: int = 64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Get timestep sequence for inference (denoising).
    
    Returns timesteps from T to 1 (excluding 0) for iterative denoising.
    
    Args:
        num_steps: Number of diffusion steps T (default 64)
        device: Device to place tensor on
    
    Returns:
        Timesteps tensor of shape (num_steps,) from T/T to 1/T
    """
    # Timesteps: T, T-1, ..., 1 normalized to [0, 1]
    steps = torch.arange(num_steps, 0, -1, device=device, dtype=torch.float32)
    return steps / num_steps


def get_remasking_threshold(
    confidence: torch.Tensor,
    t: int,
    total_steps: int,
) -> torch.Tensor:
    """
    Compute threshold for confidence-based remasking during inference.
    
    At high t: remask more aggressively (keep fewer tokens)
    At low t: remask less (keep more tokens)
    
    Args:
        confidence: Confidence scores of shape (batch_size, seq_len)
        t: Current timestep (1 to T)
        total_steps: Total number of steps T
    
    Returns:
        Per-sample threshold tensor of shape (batch_size,)
    """
    keep_ratio = t / total_steps
    
    # Compute percentile threshold per sample
    batch_size = confidence.shape[0]
    thresholds = []
    for i in range(batch_size):
        # Get threshold at keep_ratio percentile
        threshold = torch.quantile(confidence[i], keep_ratio)
        thresholds.append(threshold)
    
    return torch.stack(thresholds)


def cosine_schedule(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """
    Alternative cosine schedule (for experimentation).
    
    αₜ = cos²((t + s) / (1 + s) * π/2)
    
    Args:
        t: Timesteps tensor of shape (batch_size,) in [0, 1]
        s: Small offset to prevent singularity at t=0
    
    Returns:
        Alpha values of shape (batch_size,)
    """
    import math
    return torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2


class AbsorbingSchedule:
    """
    Encapsulates the absorbing state diffusion schedule.
    
    Provides a unified interface for:
    - Sampling timesteps
    - Computing noise levels
    - Applying noise to sequences
    """
    
    def __init__(
        self,
        mask_token_id: int,
        schedule_type: str = "linear",
        min_t: float = 0.0,
        max_t: float = 1.0,
    ):
        """
        Initialize the schedule.
        
        Args:
            mask_token_id: Token ID for [MASK]
            schedule_type: "linear" or "cosine"
            min_t: Minimum timestep for training
            max_t: Maximum timestep for training
        """
        self.mask_token_id = mask_token_id
        self.schedule_type = schedule_type
        self.min_t = min_t
        self.max_t = max_t
        
        if schedule_type == "linear":
            self._alpha_fn = get_alpha_schedule
        elif schedule_type == "cosine":
            self._alpha_fn = cosine_schedule
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps for training."""
        return sample_timesteps(batch_size, device, self.min_t, self.max_t)
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha (noise level) for timesteps."""
        return self._alpha_fn(t)
    
    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        """Get mask ratio for timesteps."""
        return 1.0 - self._alpha_fn(t)
    
    def add_noise(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        text_region_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise (masking) to input tokens.
        
        Args:
            input_ids: Clean token IDs (batch_size, seq_len)
            t: Timesteps (batch_size,)
            text_region_mask: Boolean mask for text region (batch_size, seq_len)
            generator: Optional random generator
        
        Returns:
            x_t: Noised tokens
            noise_mask: Boolean mask of noised positions
        """
        return apply_absorbing_noise(
            input_ids=input_ids,
            t=t,
            mask_token_id=self.mask_token_id,
            text_region_mask=text_region_mask,
            generator=generator,
        )
    
    def __repr__(self) -> str:
        return (
            f"AbsorbingSchedule(mask_token_id={self.mask_token_id}, "
            f"schedule={self.schedule_type}, t_range=[{self.min_t}, {self.max_t}])"
        )
