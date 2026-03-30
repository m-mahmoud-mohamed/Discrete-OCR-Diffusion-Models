"""
DiffuQwen-VL Inference Script.

Generate markdown from document images using diffusion-based OCR.

Usage:
    python infer.py --image path/to/image.png --output output.md
    python infer.py --pdf path/to/doc.pdf --output output.md
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

from diffu import DiffuQwenSampler, sample_with_kv_cache, sample

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


def load_model(
    checkpoint_path: str,
    base_model_path: str = BASE_MODEL,
    device: str = "cuda",
) -> tuple:
    """
    Load DiffuQwen model from checkpoint.
    
    Args:
        checkpoint_path: Path to LoRA checkpoint
        base_model_path: Path to base Qwen2.5-VL model
        device: Target device
    
    Returns:
        Tuple of (model, processor, tokenizer)
    """
    logger.info(f"Loading base model from {base_model_path}")
    
    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )
    
    # Load LoRA weights
    if checkpoint_path:
        logger.info(f"Loading LoRA weights from {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()  # Merge for faster inference
    
    model.eval()
    
    return model, processor, tokenizer


def load_image(path: str) -> Image.Image:
    """Load image from file or convert from PDF."""
    path = Path(path)
    
    if path.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(path), dpi=200, first_page=1, last_page=1)
            return images[0] if images else None
        except ImportError:
            raise RuntimeError("pdf2image required for PDF loading")
    else:
        return Image.open(path).convert("RGB")


def generate(
    model,
    processor,
    tokenizer,
    images: List[Image.Image],
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 1024,
    num_steps: int = 32,
    temperature: float = 0.5,
    top_p: float = 0.95,
    top_k: int = 50,
    cfg_weight: float = 1.5,
    use_diffusion: bool = True,
    use_cache: bool = False,
    save_timing: bool = True,
    visualize: bool = False,
    device: str = "cuda",
) -> Tuple[List[str], List[float], List[List[str]]]:
    """
    Generate markdown from images.
    
    Args:
        model: DiffuQwen model
        processor: Qwen processor
        tokenizer: Qwen tokenizer
        images: List of PIL images
        prompt: Generation prompt
        max_new_tokens: Maximum tokens to generate
        num_steps: Diffusion steps (ignored if use_diffusion=False)
        temperature: Sampling temperature (<=0 for deterministic)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        cfg_weight: CFG weight
        use_diffusion: Whether to use diffusion sampling
        use_cache: Whether to use KV caching (faster inference)
        save_timing: Whether to save timing metadata
        visualize: Whether to save intermediate diffusion states
        device: Target device
    
    Returns:
        Tuple of (outputs, inference_times, intermediates)
    """
    
    start_time = time.time()
    inference_times = []
    all_intermediates = []
    if use_diffusion:
        # Use diffusion sampler
        if use_cache:
            # Use KV-cached sampling for faster inference
            logger.warning("Using KV-cached diffusion sampling - WARNING: This may produce lower quality output!")
            logger.warning("KV cache is experimental and may not work well with iterative diffusion unmasking.")
            outputs = []
            for i, image in enumerate(images):
                img_start = time.time()
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
                
                from diffu import sample_with_kv_cache
                generated_tokens, _ = sample_with_kv_cache(
                    model=model,
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs.get("image_grid_thw"),
                    prompt_input_ids=inputs["input_ids"],
                    mask_token_id=MASK_TOKEN_ID,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    num_steps=num_steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    device=torch.device(device),
                )
                output = tokenizer.decode(
                    generated_tokens[0],
                    skip_special_tokens=True,
                )
                outputs.append(output)
                img_time = time.time() - img_start
                inference_times.append(img_time)
                logger.info(f"Image {i+1}/{len(images)} generated in {img_time:.2f}s")
            total_time = time.time() - start_time
            logger.info(f"Total generation time: {total_time:.2f}s (avg: {total_time/len(images):.2f}s per image)")
            return outputs, inference_times, []
        else:
            # Use standard diffusion sampler
            sampler = DiffuQwenSampler(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                mask_token_id=MASK_TOKEN_ID,
                device=torch.device(device),
            )
            
            outputs, all_intermediates = sampler.generate(
                images=images,
                prompts=[prompt] * len(images),
                max_new_tokens=max_new_tokens,
                num_steps=num_steps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                cfg_weight=cfg_weight,
                save_intermediates=visualize,
            )
            total_time = time.time() - start_time
            avg_time = total_time / len(images)
            inference_times = [avg_time] * len(images)
            logger.info(f"Total generation time: {total_time:.2f}s (avg: {avg_time:.2f}s per image)")
            return outputs, inference_times, all_intermediates
    else:
        # Use standard autoregressive generation
        outputs = []
        
        for i, image in enumerate(images):
            img_start = time.time()
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
                # Use greedy decoding if temperature <= 0
                if temperature <= 0:
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                else:
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                    )
            
            output = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            outputs.append(output)
            img_time = time.time() - img_start
            inference_times.append(img_time)
            logger.info(f"Image {i+1}/{len(images)} generated in {img_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Total generation time: {total_time:.2f}s (avg: {total_time/len(images):.2f}s per image)")
        return outputs, inference_times, []


def main():
    parser = argparse.ArgumentParser(description="DiffuQwen-VL Inference")
    
    # Input
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--pdf", type=str, help="Path to input PDF")
    parser.add_argument("--images_dir", type=str, help="Directory of images")
    
    # Output
    parser.add_argument("--output", type=str, default="output.md", help="Output file")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, help="Base model path")
    
    # Generation
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Generation prompt")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens")
    parser.add_argument("--num_steps", type=int, default=64, help="Diffusion steps")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature (0.5 optimal)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k")
    parser.add_argument("--cfg_weight", type=float, default=1.5, help="CFG weight")
    parser.add_argument("--no_diffusion", action="store_true", help="Use AR instead of diffusion")
    parser.add_argument("--use_cache", action="store_true", help="Use KV caching for faster inference (experimental, may reduce quality)")
    parser.add_argument("--save_timing", action="store_true", help="Save timing metadata to JSON files")
    parser.add_argument("--visualize", action="store_true", help="Save intermediate diffusion states for visualization")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Load model
    model, processor, tokenizer = load_model(
        checkpoint_path=args.checkpoint,
        base_model_path=args.base_model,
        device=args.device,
    )
    
    # Collect images
    images = []
    image_paths = []
    
    if args.image:
        images.append(load_image(args.image))
        image_paths.append(args.image)
    elif args.pdf:
        images.append(load_image(args.pdf))
        image_paths.append(args.pdf)
    elif args.images_dir:
        images_dir = Path(args.images_dir)
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.pdf"]:
            for path in images_dir.glob(ext):
                images.append(load_image(str(path)))
                image_paths.append(str(path))
    else:
        parser.error("Must provide --image, --pdf, or --images_dir")
    
    logger.info(f"Processing {len(images)} image(s)")
    
    # Generate
    outputs, inference_times, intermediates = generate(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        images=images,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        num_steps=args.num_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        cfg_weight=args.cfg_weight,
        use_diffusion=not args.no_diffusion,
        use_cache=args.use_cache,
        save_timing=args.save_timing,
        visualize=args.visualize,
        device=args.device,
    )
    
    # Save outputs
    if len(outputs) == 1:
        output_path = Path(args.output)
        output_path.write_text(outputs[0])
        
        if args.save_timing:
            meta_path = output_path.with_suffix('.json')
            meta = {
                "inference_time": inference_times[0],
                "num_tokens": len(outputs[0].split()),
                "tokens_per_second": len(outputs[0].split()) / inference_times[0] if inference_times[0] > 0 else 0,
                "mode": "AR" if args.no_diffusion else "diffusion",
                "use_cache": args.use_cache,
            }
            meta_path.write_text(json.dumps(meta, indent=2))
        
        if args.visualize and intermediates and len(intermediates) > 0:
            viz_path = output_path.with_stem(output_path.stem + "_intermediates").with_suffix('.json')
            viz_data = {"intermediates": intermediates[0]}
            viz_path.write_text(json.dumps(viz_data, indent=2))
            logger.info(f"Saved {len(intermediates[0])} intermediate states to {viz_path}")
        
        logger.info(f"Saved output to {output_path} ({inference_times[0]:.2f}s)")
    else:
        output_dir = Path(args.output_dir or "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (path, output, inf_time) in enumerate(zip(image_paths, outputs, inference_times)):
            output_name = Path(path).stem + ".md"
            output_path = output_dir / output_name
            output_path.write_text(output)
            
            if args.save_timing:
                meta_path = output_dir / (Path(path).stem + "_meta.json")
                meta = {
                    "source_file": path,
                    "inference_time": inf_time,
                    "num_tokens": len(output.split()),
                    "tokens_per_second": len(output.split()) / inf_time if inf_time > 0 else 0,
                    "mode": "AR" if args.no_diffusion else "diffusion",
                    "use_cache": args.use_cache,
                }
                meta_path.write_text(json.dumps(meta, indent=2))
                logger.info(f"Saved {output_path} ({inf_time:.2f}s, {meta['tokens_per_second']:.1f} tokens/s)")
            else:
                logger.info(f"Saved {output_path} ({inf_time:.2f}s)")
            
            if args.visualize and intermediates and idx < len(intermediates) and len(intermediates[idx]) > 0:
                viz_path = output_dir / (Path(path).stem + "_intermediates.json")
                viz_data = {"intermediates": intermediates[idx]}
                viz_path.write_text(json.dumps(viz_data, indent=2))
                logger.info(f"Saved {len(intermediates[idx])} intermediate states to {viz_path}")


if __name__ == "__main__":
    main()
