import os
import argparse
import math
from pathlib import Path
import torch
from PIL import Image
import copy
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def main():
    parser = argparse.ArgumentParser(description="Parallel Inference Worker for olmOCR Benchmark")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Worker ID (0, 1, 2, 3...)")
    parser.add_argument("--num-chunks", type=int, default=1, help="Total number of workers/GPUs")
    args = parser.parse_args()

    # --- Configuration ---
    pretrained = "/path/to/lavida-checkpoint"  # e.g., checkpoints/lavida-stage2-olmocr-opt-nopool/checkpoint-10750
    model_name = "llava_llada"
    
    # Conversation Template
    conv_template = "llada"
    
    # Prompt
    prompt = """Extract all text from this page in Markdown format. Follow these rules:
1. Preserve all headings, lists, tables, and formulas.
2. Use LaTeX for math equations (e.g., $x^2$).
3. Do not describe images; only transcribe text inside them.
4. Maintain the original reading order."""
    
    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    # FIXED: Since we use CUDA_VISIBLE_DEVICES in bash, always use cuda:0
    device = "cuda:0"
    
    # Paths
    base_img_dir = Path("/path/to/bench_images")  # olmOCR-bench rendered images
    base_output_dir = Path("./benchmark_outputs")  # Output directory for results
    
    categories = ["arxiv_math", "headers_footers", "long_tiny_text", "multi_column", "old_scans", "old_scans_math", "tables"]
    
    # --- Build Conversation ---
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # --- Vision Config ---
    vision_kwargs = dict(
        mm_vision_tower="/path/to/google-siglip-so400m-patch14-384",  # SigLIP vision encoder
        mm_resampler_type=None,
        mm_projector_type='mlp2x_gelu',
        mm_hidden_size=1152,
        use_mm_proj=True
    )
    
    # --- Load Model ---
    print(f"[Worker {args.chunk_idx}] Loading model on {device}...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device,
        vision_kwargs=vision_kwargs, torch_dtype='bfloat16'
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    print(f"[Worker {args.chunk_idx}] Model loaded successfully")
    
    # --- Collect All Images ---
    all_images = []
    for category in categories:
        img_dir = base_img_dir / category
        output_dir = base_output_dir / category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
        
        for img_path in image_files:
            output_file = output_dir / f"{img_path.stem}.md"
            all_images.append((img_path, output_file, category))
    
    # --- Split Work Among Workers ---
    total_images = len(all_images)
    chunk_size = math.ceil(total_images / args.num_chunks)
    start_idx = args.chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_images)
    
    my_images = all_images[start_idx:end_idx]
    
    print(f"[Worker {args.chunk_idx}] Processing {len(my_images)} images ({start_idx} to {end_idx})")
    
    if len(my_images) == 0:
        print(f"[Worker {args.chunk_idx}] No images to process. Exiting.")
        return
    
    # --- Stats ---
    processed = 0
    skipped = 0
    failed = []
    warmup_done = False
    
    # --- Process Images ---
    for img_path, output_file, category in tqdm(my_images, desc=f"Worker {args.chunk_idx}", position=args.chunk_idx):
        
        # DEBUG: Check skip logic
        if output_file.exists():
            print(f"[Worker {args.chunk_idx}] SKIPPING (exists): {output_file.name}")
            skipped += 1
            continue
        
        print(f"[Worker {args.chunk_idx}] PROCESSING: {output_file.name}")
        
        try:
            # Load and process image
            image = Image.open(img_path).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
            
            input_ids = tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            image_sizes = [image.size]
            
            # Warmup on first image
            if not warmup_done:
                print(f"[Worker {args.chunk_idx}] Running warmup...")
                with torch.inference_mode():
                    _ = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        do_sample=True,
                        temperature=0.2,
                        max_new_tokens=512,
                        block_length=128,
                        step_ratio=0.4,
                        tokenizer=tokenizer,
                        prefix_lm=False,
                        verbose=False,
                        schedule='shift',
                        schedule_kwargs={'shift': 1.0}
                    )
                warmup_done = True
                print(f"[Worker {args.chunk_idx}] Warmup complete.")
            
            print(f"[Worker {args.chunk_idx}] Generating...")
            # Generate
            with torch.inference_mode():
                cont = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    block_length=128,
                    step_ratio=0.4,
                    tokenizer=tokenizer,
                    prefix_lm=False,
                    verbose=False,
                    schedule='shift',
                    schedule_kwargs={'shift': 1.0}
                )
            
            print(f"[Worker {args.chunk_idx}] Decoding...")
            # Decode
            text_output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            text_output = text_output.lstrip('!')
            
            print(f"[Worker {args.chunk_idx}] SAVING TO: {output_file}")
            # Save with immediate flush to disk
            with output_file.open('w', encoding='utf-8') as f:
                f.write("---\n\n")
                f.write(text_output)
                f.flush()
                os.fsync(f.fileno())
            
            print(f"[Worker {args.chunk_idx}] ✓ SAVED SUCCESSFULLY: {output_file.name}")
            processed += 1
            
        except Exception as e:
            print(f"[Worker {args.chunk_idx}] ✗ FAILED {img_path.name}: {e}")
            tqdm.write(f"[Worker {args.chunk_idx}] Failed {img_path}: {e}")
            failed.append(str(img_path))
    
    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"[Worker {args.chunk_idx}] Processing complete!")
    print(f"Assigned:          {len(my_images)}")
    print(f"Processed:         {processed}")
    print(f"Skipped (exists):  {skipped}")
    print(f"Failed:            {len(failed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
