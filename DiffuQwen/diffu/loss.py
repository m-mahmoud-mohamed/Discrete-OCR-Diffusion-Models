"""
Diffusion Loss for DiffuQwen-VL.

Implements the masked cross-entropy loss with shift operation from DiffuLLaMA:
- SHIFT OPERATION: Maintain AR output structure, labels[i] = input_ids[i+1]
- MASKED LOSS: Only compute loss on masked positions in text region
- This allows seamless adaptation from AR to diffusion while preserving compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def shift_labels(
    input_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Shift labels left for next-token prediction (AR-style).
    
    labels[i] = input_ids[i+1] for i in [0, seq_len-2]
    labels[-1] = ignore_index
    
    This maintains compatibility with AR training structure while
    allowing diffusion-style masked prediction.
    
    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        ignore_index: Index to ignore in loss computation (default -100)
    
    Returns:
        Shifted labels of shape (batch_size, seq_len)
    """
    batch_size, seq_len = input_ids.shape
    labels = input_ids.new_full((batch_size, seq_len), ignore_index)
    
    # Shift left: labels[:, :-1] = input_ids[:, 1:]
    labels[:, :-1] = input_ids[:, 1:].clone()
    
    return labels


def compute_diffusion_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    noise_mask: torch.Tensor,
    text_region_mask: torch.Tensor,
    vocab_size: int,
    timesteps: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    use_time_reweight: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute diffusion loss with shift operation.
    
    Loss is computed only on:
    1. Positions that were masked (noise_mask=True)
    2. Positions in text region (text_region_mask=True)
    
    From DiffuLLaMA paper, the loss includes a 1/t reweighting term:
    L = (1/t) * sum_n [ delta(x_t^n, [MASK]) * CE(f(x_t)_n, x_0^{n+1}) ]
    
    This emphasizes learning at smaller timesteps (fewer masks).
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        input_ids: Original (clean) token IDs of shape (batch_size, seq_len)
        noise_mask: Boolean mask of noised positions (batch_size, seq_len)
        text_region_mask: Boolean mask of text region (batch_size, seq_len)
        vocab_size: Vocabulary size for reshaping
        timesteps: Timesteps for each sample (batch_size,), used for 1/t reweighting
        ignore_index: Index to ignore in loss (default -100)
        label_smoothing: Label smoothing factor (default 0.0)
        use_time_reweight: Whether to apply 1/t reweighting (default True)
    
    Returns:
        loss: Scalar loss value
        num_tokens: Number of tokens contributing to loss
    """
    batch_size, seq_len = input_ids.shape
    
    # Shift labels for next-token prediction
    labels = shift_labels(input_ids, ignore_index)
    
    # Mask out non-text regions
    labels[~text_region_mask] = ignore_index
    
    # Shift noise_mask to align with shifted labels
    # If position i was masked, we predict token at position i+1
    shifted_noise_mask = torch.zeros_like(noise_mask)
    shifted_noise_mask[:, :-1] = noise_mask[:, :-1]
    
    # Only compute loss on masked positions
    # Create final mask: positions that were masked AND are in text region
    loss_mask = shifted_noise_mask & text_region_mask
    
    # Set non-loss positions to ignore_index
    labels[~loss_mask] = ignore_index
    
    # Compute cross-entropy loss per sample
    if use_time_reweight and timesteps is not None:
        # Per-sample loss for 1/t reweighting
        loss_per_position = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view(batch_size, seq_len)
        
        # Sum loss per sample
        loss_per_sample = loss_per_position.sum(dim=1)  # (batch_size,)
        
        # Apply 1/t reweighting (with small epsilon to avoid division by zero)
        eps = 1e-6
        time_weights = 1.0 / (timesteps.clamp(min=eps))  # (batch_size,)
        weighted_loss = (loss_per_sample * time_weights).sum()
        
        # Count tokens for normalization
        num_tokens = loss_mask.sum()
        
        # Normalize by total tokens (across batch)
        if num_tokens > 0:
            loss = weighted_loss / num_tokens
        else:
            loss = weighted_loss
    else:
        # Standard loss without time reweighting
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=ignore_index,
            reduction="sum",
            label_smoothing=label_smoothing,
        )
        
        # Count number of tokens for normalization
        num_tokens = loss_mask.sum()
        
        # Normalize by number of masked tokens
        if num_tokens > 0:
            loss = loss / num_tokens
    
    return loss, num_tokens


def compute_diffusion_loss_no_shift(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    noise_mask: torch.Tensor,
    text_region_mask: torch.Tensor,
    vocab_size: int,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute diffusion loss WITHOUT shift operation.
    
    Alternative formulation where we predict the token at the masked position
    directly (pure denoising, not next-token prediction).
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        input_ids: Original (clean) token IDs of shape (batch_size, seq_len)
        noise_mask: Boolean mask of noised positions (batch_size, seq_len)
        text_region_mask: Boolean mask of text region (batch_size, seq_len)
        vocab_size: Vocabulary size for reshaping
        ignore_index: Index to ignore in loss (default -100)
        label_smoothing: Label smoothing factor (default 0.0)
    
    Returns:
        loss: Scalar loss value
        num_tokens: Number of tokens contributing to loss
    """
    batch_size, seq_len = input_ids.shape
    
    # Labels are just the original tokens (no shift)
    labels = input_ids.clone()
    
    # Create loss mask: masked positions in text region
    loss_mask = noise_mask & text_region_mask
    
    # Set non-loss positions to ignore_index
    labels[~loss_mask] = ignore_index
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
        label_smoothing=label_smoothing,
    )
    
    # Count number of tokens
    num_tokens = loss_mask.sum()
    
    # Normalize
    if num_tokens > 0:
        loss = loss / num_tokens
    
    return loss, num_tokens


class DiffusionLoss(nn.Module):
    """
    Module wrapper for diffusion loss computation.
    
    Handles shift operation, masking, and normalization.
    Implements 1/t reweighting from DiffuLLaMA paper.
    """
    
    def __init__(
        self,
        vocab_size: int,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        use_shift: bool = True,
        use_time_reweight: bool = True,
    ):
        """
        Initialize diffusion loss module.
        
        Args:
            vocab_size: Vocabulary size
            ignore_index: Index to ignore in loss computation
            label_smoothing: Label smoothing factor
            use_shift: Whether to use shift operation (AR-style)
            use_time_reweight: Whether to apply 1/t reweighting from paper
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.use_shift = use_shift
        self.use_time_reweight = use_time_reweight
    
    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        noise_mask: torch.Tensor,
        text_region_mask: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute diffusion loss.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            input_ids: Original token IDs (batch_size, seq_len)
            noise_mask: Boolean mask of noised positions (batch_size, seq_len)
            text_region_mask: Boolean mask of text region (batch_size, seq_len)
            timesteps: Timesteps for 1/t reweighting (batch_size,)
        
        Returns:
            loss: Scalar loss value
            num_tokens: Number of tokens in loss computation
        """
        if self.use_shift:
            return compute_diffusion_loss(
                logits=logits,
                input_ids=input_ids,
                noise_mask=noise_mask,
                text_region_mask=text_region_mask,
                vocab_size=self.vocab_size,
                timesteps=timesteps,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
                use_time_reweight=self.use_time_reweight,
            )
        else:
            return compute_diffusion_loss_no_shift(
                logits=logits,
                input_ids=input_ids,
                noise_mask=noise_mask,
                text_region_mask=text_region_mask,
                vocab_size=self.vocab_size,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            )
    
    def __repr__(self) -> str:
        return (
            f"DiffusionLoss(vocab_size={self.vocab_size}, "
            f"ignore_index={self.ignore_index}, "
            f"label_smoothing={self.label_smoothing}, "
            f"use_shift={self.use_shift}, "
            f"use_time_reweight={self.use_time_reweight})"
        )


def compute_per_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Compute per-token cross-entropy loss (for analysis/debugging).
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)
        vocab_size: Vocabulary size
    
    Returns:
        Per-token loss of shape (batch_size, seq_len)
    """
    batch_size, seq_len = labels.shape
    
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction="none",
    )
    
    return loss.view(batch_size, seq_len)


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute accuracy on masked positions.
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)
        mask: Boolean mask of positions to evaluate (batch_size, seq_len)
        ignore_index: Index to ignore (default -100)
    
    Returns:
        accuracy: Accuracy as a scalar
        num_correct: Number of correct predictions
    """
    # Get predictions
    preds = logits.argmax(dim=-1)
    
    # Create valid mask (in mask and not ignore_index)
    valid_mask = mask & (labels != ignore_index)
    
    # Count correct predictions
    correct = (preds == labels) & valid_mask
    num_correct = correct.sum()
    num_total = valid_mask.sum()
    
    # Compute accuracy
    if num_total > 0:
        accuracy = num_correct.float() / num_total.float()
    else:
        accuracy = torch.tensor(0.0, device=logits.device)
    
    return accuracy, num_correct
