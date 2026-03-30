"""
predict_ocr.py — Single-GPU batch OCR inference for olmOCR-bench categories.

Processes all 7 benchmark categories sequentially on one GPU, saving per-image
markdown outputs. Skips already-processed images (resume-safe).

Usage:
    python predict_ocr.py \
        --checkpoint /path/to/lavida-checkpoint \
        --vision_tower /path/to/google-siglip-so400m-patch14-384 \
        --bench_images  /path/to/olmOCR-bench/bench_data/images \
        --output_dir    /path/to/outputs
"""

import os
import copy
import time
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True,
                   help="Path to fine-tuned LaViDa-OCR checkpoint directory")
    p.add_argument("--vision_tower", required=True,
                   help="Path to google/siglip-so400m-patch14-384 weights")
    p.add_argument("--bench_images", required=True,
                   help="Root directory containing per-category image folders")
    p.add_argument("--output_dir",   required=True,
                   help="Root directory for markdown output files")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature",  type=float, default=0.2)
    p.add_argument("--step_ratio",   type=float, default=0.4,
                   help="Fraction of max steps for diffusion (~25 steps at 0.4)")
    p.add_argument("--block_length", type=int, default=64)
    return p.parse_args()


CATEGORIES = [
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "tables",
]


def main():
    args = parse_args()

    model_name = "llava_llada"
    conv_template = "llada"

    question = DEFAULT_IMAGE_TOKEN + (
        "\nExtract all text from this document page in markdown format, "
        "preserving structure, tables, formulas, and reading order."
    )

    # Build prompt template
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    vision_kwargs = dict(
        mm_vision_tower=args.vision_tower,
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        use_mm_proj=True,
    )

    print("Loading model...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.checkpoint, None, model_name,
        device_map=args.device,
        vision_kwargs=vision_kwargs,
        torch_dtype="bfloat16",
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    print("Model loaded successfully")

    base_img_dir = Path(args.bench_images)
    base_output_dir = Path(args.output_dir)

    total_images = 0
    processed_images = 0
    skipped_images = 0
    failed_images = []
    warmup_done = False

    for category in tqdm(CATEGORIES, desc="Categories"):
        img_dir = base_img_dir / category
        output_dir = base_output_dir / category
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = (
            list(img_dir.glob("*.png"))
            + list(img_dir.glob("*.jpg"))
            + list(img_dir.glob("*.jpeg"))
        )

        for img_path in tqdm(image_files, desc=f"Processing {category}", leave=False):
            total_images += 1

            output_file = output_dir / f"{img_path.stem}.md"
            if output_file.exists():
                skipped_images += 1
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)
                image_tensor = [
                    t.to(dtype=torch.bfloat16, device=args.device)
                    for t in image_tensor
                ]

                input_ids = tokenizer_image_token(
                    prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(args.device)
                image_sizes = [image.size]

                # Warmup pass (full steps, no sampling)
                if not warmup_done:
                    model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=args.max_new_tokens,
                        block_length=args.block_length,
                        step_ratio=1.0,
                        tokenizer=tokenizer,
                        prefix_lm=True,
                        verbose=False,
                    )
                    warmup_done = True

                cont = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    block_length=args.block_length,
                    step_ratio=args.step_ratio,
                    tokenizer=tokenizer,
                    prefix_lm=True,
                    verbose=False,
                    schedule="shift",
                )

                text_output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                text_output = text_output.lstrip("!")

                with output_file.open("w", encoding="utf-8") as f:
                    f.write("---\n\n")
                    f.write(text_output)

                processed_images += 1

            except Exception as e:
                tqdm.write(f"Failed: {img_path}: {e}")
                failed_images.append(str(img_path))

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total images:      {total_images}")
    print(f"Processed:         {processed_images}")
    print(f"Skipped (exists):  {skipped_images}")
    print(f"Failed:            {len(failed_images)}")
    if failed_images:
        print("\nFailed images:")
        for f in failed_images:
            print(f"  - {f}")
    print(f"{'='*60}")
    print(f"Output saved to: {base_output_dir}")


if __name__ == "__main__":
    main()
