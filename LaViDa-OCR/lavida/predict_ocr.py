import os
from pathlib import Path
import torch
from PIL import Image
import copy
import time
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# Setup
# pretrained = "/mnt/lustre-grete/projects/nii00224/mahmoud/LaViDa/lavida-ckpts/lavida-llada-hd"
pretrained = "/mnt/lustre-grete/projects/nii00224/mahmoud/LaViDa/checkpoints/lavida-stage2-olmocr-full/checkpoint-12500"
model_name = "llava_llada"
device = "cuda"
device_map = "cuda:0"
conv_template = "llada"

question = DEFAULT_IMAGE_TOKEN + """\nExtract all text from this document page in markdown format, preserving structure, tables, formulas, and reading order."""

# Build prompt
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

# Vision config
vision_kwargs = dict(
    mm_vision_tower="/mnt/lustre-grete/projects/nii00224/mahmoud/LaViDa/lavida-ckpts/google-siglip-so400m-patch14-384",
    mm_resampler_type=None,
    mm_projector_type='mlp2x_gelu',
    mm_hidden_size=1152,
    use_mm_proj=True
)

# Load model
print("Loading model...")
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map,
    vision_kwargs=vision_kwargs, torch_dtype='bfloat16'
)
model.eval()
model.tie_weights()
model.to(torch.bfloat16)
print("Model loaded successfully")

# Paths
base_img_dir = Path("/mnt/lustre-grete/projects/nii00224/mahmoud/olmOCR-bench/bench_data/images")
base_output_dir = Path("/mnt/lustre-grete/projects/nii00224/mahmoud/olmOCR-bench/bench_data/lavida_outputs_12500")

categories = ["arxiv_math", "headers_footers", "long_tiny_text", "multi_column", "old_scans", "old_scans_math", "tables"]

total_images = 0
processed_images = 0
skipped_images = 0
failed_images = []

# Warmup (optional, on first image)
warmup_done = False

for category in tqdm(categories, desc="Categories"):
    img_dir = base_img_dir / category
    output_dir = base_output_dir / category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
    
    for img_path in tqdm(image_files, desc=f"Processing {category}", leave=False):
        total_images += 1

        # Decide output path and skip if already done
        output_file = output_dir / f"{img_path.stem}.md"
        if output_file.exists():
            skipped_images += 1
            continue

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
                _ = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=2048,
                    block_length=64,
                    step_ratio=1.0,
                    tokenizer=tokenizer,
                    prefix_lm=True,
                    verbose=False,
                )
                warmup_done = True
            
            # Generate
            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.1,
                max_new_tokens=2048,  # Increase for full text
                block_length=64,
                step_ratio=0.5,
                tokenizer=tokenizer,
                prefix_lm=True,
                verbose=False,
                schedule='shift',
            )
            
            text_output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            text_output = text_output.lstrip('!')
            
            # Save as markdown
            with output_file.open('w', encoding='utf-8') as f:
                f.write("---\n\n")
                f.write(text_output)
            
            processed_images += 1
            
        except Exception as e:
            tqdm.write(f"Failed to process {img_path}: {e}")
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
