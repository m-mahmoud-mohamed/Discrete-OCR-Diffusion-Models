"""
Batch Collation for DiffuQwen-VL Training.

Handles:
- Variable-length sequences (images and text)
- Qwen2.5-VL processor integration
- Text region masking for diffusion training
- Padding and attention mask construction
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Special token IDs (will be set from tokenizer)
DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_MASK_TOKEN_ID = 151643  # Qwen's <|extra_0|> token as MASK


def prepare_qwen_inputs(
    images: List[Image.Image],
    prompts: List[str],
    targets: List[str],
    processor: Any,
    max_length: int = 4096,
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
    add_generation_prompt: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for Qwen2.5-VL using the processor.
    
    Constructs conversation format:
    <image> + prompt -> target
    
    Args:
        images: List of PIL Images
        prompts: List of prompt strings
        targets: List of target markdown strings
        processor: Qwen2.5-VL processor
        max_length: Maximum sequence length
        mask_token_id: Token ID for [MASK]
        add_generation_prompt: Whether to add generation prompt markers
    
    Returns:
        Dictionary with model inputs
    """
    batch_size = len(images)
    
    # Build conversation messages
    conversations = []
    for prompt, target in zip(prompts, targets):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": target,
            },
        ]
        conversations.append(messages)
    
    # Apply chat template
    texts = [
        processor.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conv in conversations
    ]
    
    # Process with processor
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    return inputs


def create_text_region_mask(
    input_ids: torch.Tensor,
    image_token_id: int,
    prompt_end_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Create mask indicating text output region (where masking is allowed).
    
    The text region is everything AFTER the prompt and BEFORE padding.
    Image tokens and prompt tokens are NEVER in the text region.
    
    Args:
        input_ids: Token IDs (batch_size, seq_len)
        image_token_id: ID of image placeholder token
        prompt_end_token_id: ID marking end of prompt (e.g., assistant token)
        pad_token_id: ID of padding token
    
    Returns:
        Boolean mask (batch_size, seq_len), True for text output region
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Initialize mask as all False
    text_region_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        # Find where prompt ends (look for assistant token or similar marker)
        # This is model-specific; for Qwen, look for the assistant role marker
        
        # Find first non-pad, non-image position after prompt
        # Simple heuristic: find last occurrence of prompt_end_token_id
        prompt_end_positions = (input_ids[i] == prompt_end_token_id).nonzero(as_tuple=True)[0]
        
        if len(prompt_end_positions) > 0:
            # Start text region after prompt end
            start_pos = prompt_end_positions[-1].item() + 1
        else:
            # Fallback: start after image tokens
            image_positions = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
            if len(image_positions) > 0:
                start_pos = image_positions[-1].item() + 1
            else:
                start_pos = 0
        
        # Find where padding starts
        pad_positions = (input_ids[i] == pad_token_id).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            end_pos = pad_positions[0].item()
        else:
            end_pos = seq_len
        
        # Set text region
        if start_pos < end_pos:
            text_region_mask[i, start_pos:end_pos] = True
    
    return text_region_mask


def create_text_region_mask_simple(
    input_ids: torch.Tensor,
    image_token_counts: torch.Tensor,
    prompt_lengths: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Create text region mask using explicit counts.
    
    Text region = positions after (image_tokens + prompt_length) and before padding.
    
    Args:
        input_ids: Token IDs (batch_size, seq_len)
        image_token_counts: Number of image tokens per sample (batch_size,)
        prompt_lengths: Length of prompt in tokens per sample (batch_size,)
        pad_token_id: ID of padding token
    
    Returns:
        Boolean mask (batch_size, seq_len)
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Compute start positions
    start_positions = image_token_counts + prompt_lengths
    
    # Find end positions (where padding starts)
    is_pad = input_ids == pad_token_id
    # Get first pad position per sample
    end_positions = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
    for i in range(batch_size):
        pad_indices = is_pad[i].nonzero(as_tuple=True)[0]
        if len(pad_indices) > 0:
            end_positions[i] = pad_indices[0]
    
    # Create mask
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    text_region_mask = (positions >= start_positions.unsqueeze(1)) & (positions < end_positions.unsqueeze(1))
    
    return text_region_mask


class DiffuQwenCollator:
    """
    Collator for DiffuQwen-VL training batches.
    
    Handles:
    - Qwen processor integration
    - Text region mask creation
    - Variable-length batching
    """
    
    def __init__(
        self,
        processor: Any,
        tokenizer: Any,
        max_length: int = 4096,
        mask_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        image_token_id: Optional[int] = None,
    ):
        """
        Initialize the collator.
        
        Args:
            processor: Qwen2.5-VL processor
            tokenizer: Qwen tokenizer
            max_length: Maximum sequence length
            mask_token_id: Token ID for [MASK] (uses tokenizer's if None)
            pad_token_id: Token ID for padding (uses tokenizer's if None)
            image_token_id: Token ID for image placeholder
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Get special token IDs
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id or 0
        self.mask_token_id = mask_token_id or DEFAULT_MASK_TOKEN_ID
        
        # Qwen2.5-VL uses <|image_pad|> for image tokens
        self.image_token_id = image_token_id or tokenizer.convert_tokens_to_ids("<|image_pad|>")
        
        # Get assistant token for finding prompt end
        self.assistant_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries from dataset
        
        Returns:
            Collated batch dictionary with:
            - input_ids: Token IDs (batch_size, seq_len)
            - attention_mask: Attention mask (batch_size, seq_len)
            - pixel_values: Image tensors
            - image_grid_thw: Image grid info for M-RoPE
            - text_region_mask: Boolean mask for text region
            - image_token_counts: Number of image tokens per sample
        """
        images = [sample["image"] for sample in batch]
        prompts = [sample["prompt"] for sample in batch]
        targets = [sample["text"] for sample in batch]
        
        # Build conversation format for each sample
        conversations = []
        for prompt, target in zip(prompts, targets):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": target,
                },
            ]
            conversations.append(messages)
        
        # Apply chat template
        texts = [
            self.processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conv in conversations
        ]
        
        # Truncate text content if too long (before processing to avoid image token mismatch)
        # Estimate: ~1.5 tokens per character, reserve space for image tokens (~1500 for typical image)
        max_text_chars = int((self.max_length - 2000) / 1.5)
        truncated_targets = []
        for i, (text, target) in enumerate(zip(texts, targets)):
            if len(target) > max_text_chars:
                # Truncate the target (assistant response) to fit
                truncated_targets.append(target[:max_text_chars])
                # Rebuild the conversation text with truncated target
                texts[i] = self.processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompts[i]},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": target[:max_text_chars],
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                truncated_targets.append(target)
        
        # Process with Qwen processor (no truncation to preserve image tokens)
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        
        # Final safety truncation if still too long
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > self.max_length:
            inputs["input_ids"] = input_ids[:, :self.max_length]
            inputs["attention_mask"] = inputs["attention_mask"][:, :self.max_length]
        
        # Count image tokens per sample
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        
        image_token_counts = (input_ids == self.image_token_id).sum(dim=1)
        
        # Create text region mask
        # For Qwen, find assistant response region
        text_region_mask = self._create_text_region_mask(input_ids)
        
        # Add to inputs
        inputs["text_region_mask"] = text_region_mask
        inputs["image_token_counts"] = image_token_counts
        inputs["mask_token_id"] = self.mask_token_id
        
        return inputs
    
    def _create_text_region_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create text region mask for Qwen2.5-VL format.
        
        The text region is the assistant's response content.
        Image tokens are NEVER included in the text region.
        
        Qwen2.5-VL chat template:
        <|im_start|>system\n...<|im_end|>
        <|im_start|>user\n<image>...prompt...<|im_end|>
        <|im_start|>assistant\n...response...<|im_end|>
        
        We need to find the LAST <|im_start|> (assistant's), not the first.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        text_region_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Get special token IDs
        im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Get newline token ID (for finding end of role header "assistant\n")
        newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
        
        for i in range(batch_size):
            ids = input_ids[i].tolist()
            
            # Find ALL <|im_start|> positions
            im_start_positions = [j for j, tid in enumerate(ids) if tid == im_start_token_id]
            
            if len(im_start_positions) == 0:
                continue
            
            # The LAST <|im_start|> is the assistant's message start
            assistant_im_start = im_start_positions[-1]
            
            # Find the newline AFTER <|im_start|> to locate actual content start
            # Pattern: <|im_start|>assistant\n  ->  content starts after \n
            assistant_content_start = None
            for j in range(assistant_im_start + 1, min(assistant_im_start + 10, seq_len)):
                if ids[j] == newline_token_id:
                    assistant_content_start = j + 1  # Start after newline
                    break
            
            # Fallback to +3 offset if newline not found
            if assistant_content_start is None:
                assistant_content_start = assistant_im_start + 3
            
            # Find where assistant content ends
            assistant_content_end = seq_len  # Default to end
            
            # Look for <|im_end|> or padding after assistant_content_start
            for j in range(assistant_content_start, seq_len):
                token_id = ids[j]
                if token_id == im_end_token_id or token_id == self.pad_token_id:
                    assistant_content_end = j
                    break
            
            # Mark the assistant content region
            if assistant_content_start < assistant_content_end:
                text_region_mask[i, assistant_content_start:assistant_content_end] = True
        
        # CRITICAL: Explicitly exclude image tokens from the mask
        # Image tokens should NEVER be masked, as this would corrupt
        # the image-text alignment in the model
        is_image_token = (input_ids == self.image_token_id)
        text_region_mask = text_region_mask & ~is_image_token
        
        # Also exclude special tokens that should never be masked
        is_special = (input_ids == self.pad_token_id) | \
                     (input_ids == im_start_token_id) | \
                     (input_ids == im_end_token_id)
        text_region_mask = text_region_mask & ~is_special
        
        return text_region_mask


class SimpleCollator:
    """
    Simple collator that works without processor dependency.
    
    For testing and debugging purposes.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 4096,
    ):
        """
        Initialize simple collator.
        
        Args:
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch without image processing.
        
        Returns raw data for debugging.
        """
        return {
            "images": [sample["image"] for sample in batch],
            "texts": [sample["text"] for sample in batch],
            "prompts": [sample["prompt"] for sample in batch],
            "source_paths": [sample["source_path"] for sample in batch],
        }
