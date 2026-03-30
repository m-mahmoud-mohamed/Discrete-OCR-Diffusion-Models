"""
Evaluation Script for DiffuQwen-VL.

Compare DiffuQwen against AR baseline on OCR metrics:
- Character Error Rate (CER)
- Word Error Rate (WER)
- BLEU score
- Inference speed (tokens/second)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

from qwen import OLMoCRDataset
from diffu import DiffuQwenSampler

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Constants
# olmOCR-style prompt (from build_no_anchoring_v4_yaml_prompt without YAML footer)
DEFAULT_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LaTeX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)"
)
MASK_TOKEN_ID = 151643
BASE_MODEL = "/path/to/olmocr-model"
DATASET_ROOT = "/path/to/olmocr-dataset"


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER).
    
    CER = edit_distance(ref, hyp) / len(ref)
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    
    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER).
    
    WER = edit_distance(ref_words, hyp_words) / len(ref_words)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Use word-level Levenshtein
    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute all OCR metrics.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        Dictionary of metrics
    """
    # Normalize whitespace
    ref_clean = " ".join(reference.split())
    hyp_clean = " ".join(hypothesis.split())
    
    return {
        "cer": character_error_rate(ref_clean, hyp_clean),
        "wer": word_error_rate(ref_clean, hyp_clean),
        "ref_len": len(ref_clean),
        "hyp_len": len(hyp_clean),
    }


def evaluate_model(
    model,
    processor,
    tokenizer,
    dataset: OLMoCRDataset,
    use_diffusion: bool = True,
    num_steps: int = 64,
    temperature: float = 0.7,
    max_samples: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate model on dataset.
    
    Args:
        model: DiffuQwen or Qwen model
        processor: Qwen processor
        tokenizer: Qwen tokenizer
        dataset: Evaluation dataset
        use_diffusion: Whether to use diffusion sampling
        num_steps: Diffusion steps
        temperature: Sampling temperature
        max_samples: Maximum samples to evaluate
        device: Target device
    
    Returns:
        Dictionary of aggregated metrics
    """
    model.eval()
    
    all_metrics = []
    total_time = 0.0
    total_tokens = 0
    
    num_samples = min(len(dataset), max_samples)
    
    if use_diffusion:
        sampler = DiffuQwenSampler(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            mask_token_id=MASK_TOKEN_ID,
            device=torch.device(device),
        )
    
    for idx in tqdm(range(num_samples), desc="Evaluating"):
        sample = dataset[idx]
        image = sample["image"]
        reference = sample["text"]
        prompt = sample["prompt"]
        
        # Time generation
        start_time = time.time()
        
        if use_diffusion:
            outputs = sampler.generate(
                images=[image],
                prompts=[prompt],
                num_steps=num_steps,
                temperature=temperature,
            )
            hypothesis = outputs[0]
        else:
            # Autoregressive generation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=temperature,
                    do_sample=True,
                )
            
            hypothesis = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
        
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        total_tokens += len(hypothesis)
        
        # Compute metrics
        metrics = compute_metrics(reference, hypothesis)
        all_metrics.append(metrics)
    
    # Aggregate metrics
    avg_cer = sum(m["cer"] for m in all_metrics) / len(all_metrics)
    avg_wer = sum(m["wer"] for m in all_metrics) / len(all_metrics)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "cer": avg_cer,
        "wer": avg_wer,
        "tokens_per_second": tokens_per_second,
        "total_time": total_time,
        "num_samples": num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DiffuQwen-VL")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, help="Base model")
    
    # Data
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT, help="Dataset root")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples")
    
    # Generation
    parser.add_argument("--num_steps", type=int, default=64, help="Diffusion steps")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--compare_ar", action="store_true", help="Compare with AR baseline")
    
    # Output
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output file")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Load model
    logger.info("Loading model...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.device,
    )
    
    if args.checkpoint:
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()
    
    model.eval()
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = OLMoCRDataset(
        root_dir=args.dataset_root,
        split="eval",
        max_samples=args.max_samples,
    )
    
    results = {}
    
    # Evaluate diffusion
    logger.info("Evaluating diffusion model...")
    diffusion_metrics = evaluate_model(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        dataset=dataset,
        use_diffusion=True,
        num_steps=args.num_steps,
        temperature=args.temperature,
        max_samples=args.max_samples,
        device=args.device,
    )
    results["diffusion"] = diffusion_metrics
    
    logger.info(f"Diffusion - CER: {diffusion_metrics['cer']:.4f}, "
                f"WER: {diffusion_metrics['wer']:.4f}, "
                f"Tokens/s: {diffusion_metrics['tokens_per_second']:.1f}")
    
    # Evaluate AR baseline
    if args.compare_ar:
        logger.info("Evaluating AR baseline...")
        ar_metrics = evaluate_model(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            dataset=dataset,
            use_diffusion=False,
            temperature=args.temperature,
            max_samples=args.max_samples,
            device=args.device,
        )
        results["autoregressive"] = ar_metrics
        
        logger.info(f"AR - CER: {ar_metrics['cer']:.4f}, "
                    f"WER: {ar_metrics['wer']:.4f}, "
                    f"Tokens/s: {ar_metrics['tokens_per_second']:.1f}")
        
        # Compute speedup
        if ar_metrics["tokens_per_second"] > 0:
            speedup = diffusion_metrics["tokens_per_second"] / ar_metrics["tokens_per_second"]
            results["speedup"] = speedup
            logger.info(f"Speedup: {speedup:.2f}x")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
