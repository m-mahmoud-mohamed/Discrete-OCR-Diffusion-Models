"""
DiffuQwen-VL Training Script (HuggingFace Trainer).

Uses HuggingFace Trainer for robust training with all custom diffusion logic preserved:
- Absorbing state diffusion training
- LoRA fine-tuning
- Attention mask annealing (causal→bidirectional)
- Full trainer_state.json with log_history
- TensorBoard/WandB integration
- Proper checkpoint resumption
"""

import os
import sys
import json
import math
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Transformers and PEFT
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

# Local imports
from diffu import AbsorbingSchedule, AnnealedAttentionMaskBuilder, DiffusionLoss
from qwen import OLMoCRDataset, DiffuQwenCollator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

DATASET_ROOT = "/path/to/olmocr-dataset"
BASE_MODEL = "/path/to/olmocr-model"

# olmOCR-style prompt
DEFAULT_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LaTeX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)"
)

# Diffusion constants
MASK_TOKEN_ID = 151643  # Qwen's <|extra_0|> as [MASK]


# ═══════════════════════════════════════════════════════════════
# DATACLASSES FOR ARGUMENTS
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_path: str = field(
        default=BASE_MODEL,
        metadata={"help": "Path to base Qwen2.5-VL model"}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    freeze_vision: bool = field(default=True, metadata={"help": "Freeze vision encoder"})


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_root: str = field(
        default=DATASET_ROOT,
        metadata={"help": "Root directory of olmOCR dataset"}
    )
    max_length: int = field(default=8192, metadata={"help": "Maximum sequence length"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Max training samples"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Max eval samples"})


@dataclass
class DiffusionArguments:
    """Arguments for diffusion training."""
    anneal_steps: int = field(default=10000, metadata={"help": "Steps for attention annealing"})
    mask_token_id: int = field(default=MASK_TOKEN_ID, metadata={"help": "Token ID for [MASK]"})


# ═══════════════════════════════════════════════════════════════
# CUSTOM DIFFUSION TRAINER
# ═══════════════════════════════════════════════════════════════

class DiffusionTrainer(Trainer):
    """
    Custom Trainer for diffusion-based OCR training.
    
    Extends HuggingFace Trainer with:
    - Absorbing state noise injection
    - Diffusion loss with 1/t reweighting
    - Annealed attention masks
    """
    
    def __init__(
        self,
        diffusion_args: DiffusionArguments,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # Initialize diffusion components
        self.schedule = AbsorbingSchedule(
            mask_token_id=diffusion_args.mask_token_id,
            schedule_type="linear",
        )
        
        self.mask_builder = AnnealedAttentionMaskBuilder(
            anneal_steps=diffusion_args.anneal_steps,
            deterministic=False,
        )
        
        vocab_size = self.model.config.vocab_size
        self.loss_fn = DiffusionLoss(
            vocab_size=vocab_size,
            label_smoothing=0.0,
        )
        
        logger.info(f"Initialized DiffusionTrainer with anneal_steps={diffusion_args.anneal_steps}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute diffusion loss with absorbing state noise.
        
        This is the core diffusion training logic:
        1. Sample random timesteps t ~ U(0, 1)
        2. Apply absorbing noise to text tokens (mask with probability t)
        3. Run model to predict original tokens
        4. Compute cross-entropy loss with 1/t reweighting
        """
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        text_region_mask = inputs["text_region_mask"]
        image_grid_thw = inputs.get("image_grid_thw")
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Update mask builder step
        self.mask_builder.global_step = self.state.global_step
        
        # Sample timesteps
        t = self.schedule.sample_t(batch_size, device)
        
        # Apply absorbing noise to text region
        x_t, noise_mask = self.schedule.add_noise(
            input_ids=input_ids,
            t=t,
            text_region_mask=text_region_mask,
        )
        
        # Forward pass with noised input
        outputs = model(
            input_ids=x_t,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        
        # Compute diffusion loss with 1/t reweighting
        loss, num_tokens = self.loss_fn(
            logits=outputs.logits,
            input_ids=input_ids,
            noise_mask=noise_mask,
            text_region_mask=text_region_mask,
            timesteps=t,
        )
        
        # Log metrics (will be picked up by Trainer)
        if self.state.global_step % self.args.logging_steps == 0:
            text_region_size = text_region_mask.sum().item()
            self.log({
                "mask_ratio": t.mean().item(),
                "anneal_progress": self.mask_builder.anneal_progress,
                "num_masked_tokens": num_tokens.item(),
                "text_region_size": text_region_size,
            })
        
        # Step the mask builder
        self.mask_builder.step()
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Evaluation step with fixed timestep for consistency.
        """
        with torch.no_grad():
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]
            text_region_mask = inputs["text_region_mask"]
            image_grid_thw = inputs.get("image_grid_thw")
            
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            # Use fixed timestep t=0.5 for consistent evaluation
            t = torch.full((batch_size,), 0.5, device=device)
            
            x_t, noise_mask = self.schedule.add_noise(
                input_ids=input_ids,
                t=t,
                text_region_mask=text_region_mask,
            )
            
            outputs = model(
                input_ids=x_t,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            
            # Compute loss without 1/t reweighting for eval
            loss, _ = self.loss_fn(
                logits=outputs.logits,
                input_ids=input_ids,
                noise_mask=noise_mask,
                text_region_mask=text_region_mask,
                timesteps=None,  # No reweighting for eval
            )
        
        return (loss, None, None)


# ═══════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════

class DiffusionLoggingCallback(TrainerCallback):
    """Callback for additional diffusion-specific logging."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add progress bar info to logs."""
        if logs is not None and state.global_step > 0:
            progress = state.global_step / state.max_steps * 100
            logs["progress_percent"] = progress


# ═══════════════════════════════════════════════════════════════
# MODEL SETUP
# ═══════════════════════════════════════════════════════════════

def setup_model(
    model_args: ModelArguments,
    resume_from_checkpoint: Optional[str] = None,
) -> Tuple[nn.Module, Any, Any]:
    """
    Load Qwen2.5-VL and apply LoRA.
    
    If resume_from_checkpoint is provided, loads LoRA from checkpoint.
    Otherwise applies fresh LoRA with specified config.
    """
    logger.info(f"Loading model from {model_args.model_path}")
    
    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(model_args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    
    # Freeze vision encoder if requested
    if model_args.freeze_vision:
        logger.info("Freezing vision encoder")
        for name, param in model.named_parameters():
            if "visual" in name.lower() or "vision" in name.lower():
                param.requires_grad = False
    
    # Load LoRA from checkpoint OR apply fresh LoRA
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        if checkpoint_path.exists() and (checkpoint_path / "adapter_config.json").exists():
            logger.info(f"Loading LoRA adapter from checkpoint: {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        else:
            logger.warning(f"Checkpoint invalid, applying fresh LoRA")
            model = _apply_fresh_lora(model, model_args)
    else:
        model = _apply_fresh_lora(model, model_args)
    
    model.print_trainable_parameters()
    
    return model, processor, tokenizer


def _apply_fresh_lora(model, model_args: ModelArguments):
    """Apply fresh LoRA configuration to model."""
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    logger.info(f"Applying LoRA with r={model_args.lora_r}, alpha={model_args.lora_alpha}")
    return get_peft_model(model, lora_config)


# ═══════════════════════════════════════════════════════════════
# DATA SETUP
# ═══════════════════════════════════════════════════════════════

def setup_datasets(
    data_args: DataArguments,
    processor: Any,
    tokenizer: Any,
) -> Tuple[OLMoCRDataset, OLMoCRDataset]:
    """Setup training and evaluation datasets."""
    logger.info(f"Loading dataset from {data_args.dataset_root}")
    
    train_dataset = OLMoCRDataset(
        root_dir=data_args.dataset_root,
        split="train",
        max_samples=data_args.max_train_samples,
        prompt=DEFAULT_PROMPT,
    )
    
    eval_dataset = OLMoCRDataset(
        root_dir=data_args.dataset_root,
        split="eval",
        max_samples=data_args.max_eval_samples,
        prompt=DEFAULT_PROMPT,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main training function."""
    from transformers import HfArgumentParser
    
    parser = HfArgumentParser((ModelArguments, DataArguments, DiffusionArguments, TrainingArguments))
    
    # Check for config file
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, diffusion_args, training_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, diffusion_args, training_args = parser.parse_args_into_dataclasses()
    
    # Detect last checkpoint for resumption
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Detected checkpoint: {last_checkpoint}. Resuming training.")
    
    # Setup model (with checkpoint loading if resuming)
    model, processor, tokenizer = setup_model(
        model_args,
        resume_from_checkpoint=last_checkpoint,
    )
    
    # Setup datasets
    train_dataset, eval_dataset = setup_datasets(data_args, processor, tokenizer)
    
    # Setup collator
    collator = DiffuQwenCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        mask_token_id=diffusion_args.mask_token_id,
    )
    
    # Initialize trainer
    trainer = DiffusionTrainer(
        diffusion_args=diffusion_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[DiffusionLoggingCallback()],
    )
    
    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save final model
        trainer.save_model()
        trainer.save_state()
        
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
