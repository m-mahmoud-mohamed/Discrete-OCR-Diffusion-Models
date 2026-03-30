"""
olmocr_infer.py — Run olmOCR (autoregressive baseline) inference on olmOCR-bench.

Processes all 7 benchmark categories on a single GPU, saving per-image markdown
outputs with benchmark-compatible filenames ({stem}_pg1_repeat1.md).

Usage:
    python olmocr_infer.py \
        --model_path  /path/to/olmOCR-7B-0225-preview \
        --processor_path /path/to/Qwen2.5-VL-7B-Instruct \
        --bench_images /path/to/olmOCR-bench/bench_data/images \
        --output_dir   /path/to/olmocr_outputs
"""

import argparse
import base64
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

CATEGORIES = [
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "tables",
]


def parse_args():
    p = argparse.ArgumentParser(description="olmOCR baseline inference on olmOCR-bench")
    p.add_argument("--model_path", required=True,
                   help="Path to olmOCR model (e.g. allenai/olmOCR-7B-0225-preview)")
    p.add_argument("--processor_path", required=True,
                   help="Path to Qwen2.5-VL processor (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    p.add_argument("--bench_images", required=True,
                   help="Root directory containing per-category image folders")
    p.add_argument("--output_dir", required=True,
                   help="Root directory for markdown output files")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.1)
    return p.parse_args()


def image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    args = parse_args()

    print("Loading olmOCR model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).eval()
    processor = AutoProcessor.from_pretrained(args.processor_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    base_img_dir = Path(args.bench_images)
    base_output_dir = Path(args.output_dir)

    total_images = 0
    processed_images = 0
    skipped_images = 0
    failed_images = []

    for category in tqdm(CATEGORIES, desc="Categories"):
        img_dir = base_img_dir / category
        output_dir = base_output_dir / category
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))

        for img_path in tqdm(image_files, desc=f"Processing {category}", leave=False):
            total_images += 1

            stem = img_path.stem
            stem = stem.replace("_pg1", "").replace("_repeat1", "")
            output_filename = f"{stem}_pg1_repeat1.md"
            output_file = output_dir / output_filename

            if output_file.exists():
                skipped_images += 1
                continue

            try:
                image_base64 = image_to_base64(img_path)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

                inputs = processor(
                    text=[text], images=[main_image], padding=True, return_tensors="pt"
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}

                output = model.generate(
                    **inputs,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                )

                prompt_length = inputs["input_ids"].shape[1]
                new_tokens = output[:, prompt_length:]
                text_output = processor.tokenizer.batch_decode(
                    new_tokens, skip_special_tokens=True
                )[0]

                with output_file.open("w", encoding="utf-8") as f:
                    if not text_output.strip().startswith("---"):
                        f.write("---\n\n")
                    f.write(text_output)

                processed_images += 1

            except Exception as e:
                tqdm.write(f"Failed to process {img_path}: {e}")
                failed_images.append(str(img_path))

    print(f"\n{'='*60}")
    print("olmOCR processing complete!")
    print(f"Total images:      {total_images}")
    print(f"Processed:         {processed_images}")
    print(f"Skipped (exists):  {skipped_images}")
    print(f"Failed:            {len(failed_images)}")
    if failed_images:
        print("\nFailed images:")
        for f in failed_images[:10]:
            print(f"  - {f}")
    print(f"{'='*60}")
    print(f"Output saved to: {base_output_dir}")


if __name__ == "__main__":
    main()
