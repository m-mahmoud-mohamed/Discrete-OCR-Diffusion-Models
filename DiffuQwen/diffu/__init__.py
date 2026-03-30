"""
DiffuQwen Diffusion Components.

This module contains core diffusion algorithms:
- schedule: Absorbing state noise schedule
- attention: Annealed attention mask builder
- loss: Masked cross-entropy with shift operation
- sampler: Iterative denoising with Prefix-DLM caching
"""

from .schedule import (
    AbsorbingSchedule,
    sample_timesteps,
    get_alpha_schedule,
    get_mask_ratio,
    apply_absorbing_noise,
    get_inference_timesteps,
)
from .attention import (
    build_annealed_attention_mask,
    build_multimodal_attention_mask,
    AnnealedAttentionMaskBuilder,
)
from .loss import (
    compute_diffusion_loss,
    shift_labels,
    DiffusionLoss,
)
from .sampler import (
    sample,
    DiffuQwenSampler,
    sample_with_kv_cache,
    sample_with_prefix_caching,
)

__all__ = [
    # Schedule
    "AbsorbingSchedule",
    "sample_timesteps",
    "get_alpha_schedule",
    "get_mask_ratio",
    "apply_absorbing_noise",
    "get_inference_timesteps",
    # Attention
    "build_annealed_attention_mask",
    "build_multimodal_attention_mask",
    "AnnealedAttentionMaskBuilder",
    # Loss
    "compute_diffusion_loss",
    "shift_labels",
    "DiffusionLoss",
    # Sampler
    "sample",
    "DiffuQwenSampler",
    "sample_with_kv_cache",
    "sample_with_prefix_caching",
]
