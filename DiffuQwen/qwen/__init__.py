"""
DiffuQwen Qwen2.5-VL Components.

This module contains Qwen-specific implementations:
- data: Dataset loading for olmOCR
- collator: Batch collation with masking
- attention_patch: Custom attention mask injection
- modeling: Model wrapper with LoRA
"""

from .data import (
    OLMoCRDataset,
    OLMoCRDatasetFromList,
    find_pdf_md_pairs,
    find_image_md_pairs,
    create_train_eval_split,
    get_dataloader,
    DATASET_ROOT,
    DEFAULT_PROMPT,
)
from .collator import (
    DiffuQwenCollator,
    SimpleCollator,
    create_text_region_mask,
    create_text_region_mask_simple,
)
from .attention_patch import (
    QwenAttentionPatcher,
    DiffuQwenAttentionWrapper,
    patch_attention_mask,
    unpatch_attention_mask,
    apply_bidirectional_attention,
    create_annealed_mask_hook,
)

__all__ = [
    # Data
    "OLMoCRDataset",
    "OLMoCRDatasetFromList",
    "find_pdf_md_pairs",
    "find_image_md_pairs",
    "create_train_eval_split",
    "get_dataloader",
    "DATASET_ROOT",
    "DEFAULT_PROMPT",
    # Collator
    "DiffuQwenCollator",
    "SimpleCollator",
    "create_text_region_mask",
    "create_text_region_mask_simple",
    # Attention
    "QwenAttentionPatcher",
    "DiffuQwenAttentionWrapper",
    "patch_attention_mask",
    "unpatch_attention_mask",
    "apply_bidirectional_attention",
    "create_annealed_mask_hook",
]
