"""
Attention Patching for DiffuQwen-VL.

Patches Qwen2.5-VL's attention mechanism to support:
- Custom attention masks (annealed causal→bidirectional)
- Multimodal attention rules (image/text interactions)
- Compatibility with M-RoPE (no modifications needed)

This module provides utilities to inject custom attention masks
into Qwen2.5-VL's forward pass without modifying the original model code.
"""

import logging
from typing import Any, Callable, Optional, Tuple
from functools import wraps

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def patch_attention_mask(
    model: nn.Module,
    custom_mask_fn: Callable[[torch.Tensor, int], torch.Tensor],
    layer_names: Optional[Tuple[str, ...]] = None,
) -> nn.Module:
    """
    Patch model's attention layers to use custom attention masks.
    
    This is a high-level patching function that intercepts attention
    computations and injects custom masks.
    
    Args:
        model: Qwen2.5-VL model
        custom_mask_fn: Function (attention_mask, global_step) -> new_mask
        layer_names: Specific layer names to patch (None = all attention layers)
    
    Returns:
        Patched model (modifies in-place)
    """
    # Store original forward methods
    model._original_forwards = {}
    model._custom_mask_fn = custom_mask_fn
    model._global_step = 0
    
    # Find and patch attention layers
    for name, module in model.named_modules():
        if _is_attention_layer(module, layer_names):
            _patch_single_layer(model, name, module)
            logger.debug(f"Patched attention layer: {name}")
    
    return model


def _is_attention_layer(module: nn.Module, layer_names: Optional[Tuple[str, ...]]) -> bool:
    """Check if module is an attention layer to patch."""
    # Qwen2.5-VL uses Qwen2VLSdpaAttention or similar
    module_name = type(module).__name__.lower()
    
    if layer_names is not None:
        return any(ln.lower() in module_name for ln in layer_names)
    
    return "attention" in module_name and "self" in module_name


def _patch_single_layer(model: nn.Module, name: str, module: nn.Module):
    """Patch a single attention layer's forward method."""
    original_forward = module.forward
    model._original_forwards[name] = original_forward
    
    @wraps(original_forward)
    def patched_forward(*args, **kwargs):
        # Intercept attention_mask if provided
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            original_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = model._custom_mask_fn(
                original_mask, 
                model._global_step
            )
        return original_forward(*args, **kwargs)
    
    module.forward = patched_forward


def unpatch_attention_mask(model: nn.Module) -> nn.Module:
    """
    Remove attention patching from model.
    
    Args:
        model: Previously patched model
    
    Returns:
        Unpatched model
    """
    if not hasattr(model, "_original_forwards"):
        logger.warning("Model was not patched")
        return model
    
    for name, original_forward in model._original_forwards.items():
        # Find module by name and restore forward
        parts = name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        module.forward = original_forward
    
    del model._original_forwards
    del model._custom_mask_fn
    del model._global_step
    
    return model


class QwenAttentionPatcher:
    """
    Context manager and class for patching Qwen2.5-VL attention.
    
    Provides a cleaner interface for attention mask injection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        anneal_steps: int = 10000,
    ):
        """
        Initialize the patcher.
        
        Args:
            model: Qwen2.5-VL model to patch
            anneal_steps: Steps for causal→bidirectional annealing
        """
        self.model = model
        self.anneal_steps = anneal_steps
        self._global_step = 0
        self._patched = False
    
    @property
    def global_step(self) -> int:
        return self._global_step
    
    @global_step.setter
    def global_step(self, value: int):
        self._global_step = value
        if self._patched:
            self.model._global_step = value
    
    @property
    def anneal_progress(self) -> float:
        return min(1.0, self._global_step / self.anneal_steps)
    
    def patch(self) -> "QwenAttentionPatcher":
        """Apply attention patching."""
        if self._patched:
            logger.warning("Model already patched")
            return self
        
        def mask_modifier(mask: torch.Tensor, step: int) -> torch.Tensor:
            return self._modify_attention_mask(mask, step)
        
        patch_attention_mask(self.model, mask_modifier)
        self.model._global_step = self._global_step
        self._patched = True
        
        return self
    
    def unpatch(self) -> "QwenAttentionPatcher":
        """Remove attention patching."""
        if not self._patched:
            return self
        
        unpatch_attention_mask(self.model)
        self._patched = False
        
        return self
    
    def _modify_attention_mask(
        self,
        mask: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Modify attention mask based on annealing progress.
        
        Args:
            mask: Original attention mask
            step: Current global step
        
        Returns:
            Modified attention mask with annealing applied
        """
        progress = min(1.0, step / self.anneal_steps)
        
        if progress >= 1.0:
            # Full bidirectional - return all zeros (full attention)
            return torch.zeros_like(mask)
        
        if progress <= 0.0:
            # Pure causal - return original mask
            return mask
        
        # Annealing: probabilistically unmask upper triangle
        # Only modify where mask is -inf (blocked positions)
        is_blocked = mask < -1e9  # -inf positions
        
        # Sample random values
        rand_vals = torch.rand_like(mask.float())
        
        # Unmask positions where random < progress
        should_unmask = is_blocked & (rand_vals < progress)
        
        # Create new mask
        new_mask = mask.clone()
        new_mask[should_unmask] = 0.0
        
        return new_mask
    
    def __enter__(self):
        return self.patch()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unpatch()
        return False


def create_annealed_mask_hook(
    anneal_steps: int = 10000,
) -> Callable:
    """
    Create a hook function for attention mask annealing.
    
    This can be registered as a forward hook on attention layers.
    
    Args:
        anneal_steps: Steps for annealing
    
    Returns:
        Hook function compatible with register_forward_pre_hook
    """
    state = {"step": 0}
    
    def hook(module: nn.Module, args: Tuple, kwargs: dict) -> Tuple[Tuple, dict]:
        """Pre-forward hook to modify attention mask."""
        if "attention_mask" not in kwargs or kwargs["attention_mask"] is None:
            return args, kwargs
        
        mask = kwargs["attention_mask"]
        progress = min(1.0, state["step"] / anneal_steps)
        
        if progress >= 1.0:
            kwargs["attention_mask"] = torch.zeros_like(mask)
        elif progress > 0.0:
            is_blocked = mask < -1e9
            rand_vals = torch.rand_like(mask.float())
            should_unmask = is_blocked & (rand_vals < progress)
            new_mask = mask.clone()
            new_mask[should_unmask] = 0.0
            kwargs["attention_mask"] = new_mask
        
        return args, kwargs
    
    # Attach step updater to hook function
    hook.update_step = lambda s: state.update({"step": s})
    hook.get_step = lambda: state["step"]
    
    return hook


def get_qwen_attention_layers(model: nn.Module) -> list:
    """
    Get all attention layer modules from a Qwen model.
    
    Args:
        model: Qwen2.5-VL model
    
    Returns:
        List of (name, module) tuples for attention layers
    """
    attention_layers = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "Attention" in module_type and "self" in name.lower():
            attention_layers.append((name, module))
    
    return attention_layers


def apply_bidirectional_attention(model: nn.Module) -> nn.Module:
    """
    Force full bidirectional attention (for inference).
    
    Patches model to always use full attention mask.
    
    Args:
        model: Qwen2.5-VL model
    
    Returns:
        Patched model
    """
    def full_attention_fn(mask: torch.Tensor, step: int) -> torch.Tensor:
        return torch.zeros_like(mask)
    
    return patch_attention_mask(model, full_attention_fn)


class DiffuQwenAttentionWrapper(nn.Module):
    """
    Wrapper module that handles attention mask injection.
    
    Wraps the base Qwen model and modifies attention masks
    during forward pass based on diffusion training requirements.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        anneal_steps: int = 10000,
    ):
        """
        Initialize the wrapper.
        
        Args:
            base_model: Qwen2.5-VL model
            anneal_steps: Steps for attention annealing
        """
        super().__init__()
        self.base_model = base_model
        self.anneal_steps = anneal_steps
        self._global_step = 0
    
    @property
    def global_step(self) -> int:
        return self._global_step
    
    @global_step.setter
    def global_step(self, value: int):
        self._global_step = max(0, value)
    
    @property
    def anneal_progress(self) -> float:
        return min(1.0, self._global_step / self.anneal_steps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        custom_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with attention mask modification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Standard attention mask (will be modified)
            custom_attention_mask: Explicit custom mask (overrides annealing)
            **kwargs: Additional arguments passed to base model
        
        Returns:
            Base model outputs
        """
        # Use custom mask if provided, otherwise apply annealing
        if custom_attention_mask is not None:
            final_mask = custom_attention_mask
        elif attention_mask is not None:
            final_mask = self._apply_annealing(attention_mask)
        else:
            final_mask = None
        
        return self.base_model(
            input_ids=input_ids,
            attention_mask=final_mask,
            **kwargs,
        )
    
    def _apply_annealing(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply annealing to attention mask."""
        progress = self.anneal_progress
        
        if progress >= 1.0:
            return torch.zeros_like(mask)
        elif progress <= 0.0:
            return mask
        
        # Probabilistic unmasking
        is_blocked = mask < -1e9
        rand_vals = torch.rand_like(mask.float())
        should_unmask = is_blocked & (rand_vals < progress)
        
        new_mask = mask.clone()
        new_mask[should_unmask] = 0.0
        
        return new_mask
    
    def __getattr__(self, name: str):
        """Delegate attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
