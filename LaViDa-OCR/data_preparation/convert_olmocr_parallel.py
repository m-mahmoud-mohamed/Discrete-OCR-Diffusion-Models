"""
Parallel olmOCR to LaViDa converter - OPTIMIZED (High Quality)
- Converts PDFs to PNGs (FAST)
- Creates LaViDa format JSON
- Keeps original PDFs
"""

from pathlib import Path
import os
import json
from tqdm import tqdm
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

# Suppress PIL decompression bomb warnings
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def find_all_pdf_md_pairs(base_dir):
    """Recursively find all PDF-metadata pairs"""
    base_path = Path(base_dir)

    print("Scanning for PDF files...")
    all_pdfs = list(base_path.rglob("*.pdf"))
    print(f"✓ Found {len(all_pdfs)} PDF files")

    # Filter to those with .md files
    pairs = []
    for pdf_path in all_pdfs:
        md_path = pdf_path.with_suffix(".md")
        if md_path.exists():
            pairs.append((pdf_path, md_path))

    print(f"✓ Found {len(pairs)} PDF-MD pairs")
    return pairs


def process_single_pair(args):
    """Process one PDF-MD pair (for parallel execution) - OPTIMIZED"""
    pdf_path, md_path, images_dir, dpi = args

    try:
        # Output path for PNG
        output_path = images_dir / f"{pdf_path.stem}.png"

        # Skip if already exists
        if output_path.exists():
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
            return {
                "success": True,
                "image_path": str(output_path),
                "text": text,
                "id": pdf_path.stem,
                "skipped": True,
            }

        # Convert PDF to PNG - OPTIMIZED FOR SPEED
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt="png",
            thread_count=1,
            use_pdftocairo=True,  # Fast renderer (no quality loss)
        )

        # Save without extra optimization for speed (PNG is still lossless)
        images[0].save(output_path, "PNG", optimize=False)

        # Read text metadata
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        return {
            "success": True,
            "image_path": str(output_path),
            "text": text,
            "id": pdf_path.stem,
            "skipped": False,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "pdf_path": str(pdf_path),
        }


def parallel_convert_all(pairs, images_dir, dpi=150, n_workers=32):
    """Convert all PDFs in parallel"""

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments
    args_list = [(pdf, md, images_dir, dpi) for pdf, md in pairs]

    results = []
    failed = 0
    skipped = 0
    converted = 0

    print(f"\nConverting with {n_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_pair, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting PDFs"):
            result = future.result()

            if result["success"]:
                results.append(result)
                if result.get("skipped"):
                    skipped += 1
                else:
                    converted += 1
            else:
                failed += 1

    print(f"\n✓ Converted: {converted}")
    print(f"⊙ Skipped (already exist): {skipped}")
    print(f"✗ Failed: {failed}")

    return [r for r in results if r["success"]]


def create_lavida_json(results, lavida_data_root):
    """Create LaViDa format JSON files"""

    train_samples = []
    eval_samples = []

    for result in results:
        full_path = Path(result["image_path"])
        parent_dirs = str(full_path.parent)

        is_train = "train" in parent_dirs
        is_eval = "eval" in parent_dirs

        # Path relative to LaViDa repo root (as expected by training code)
        relative_path = f"data/olmocr/images/{full_path.name}"

        sample = {
            "id": result["id"],
            "image": relative_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nExtract all text from this document page in markdown format, "
                             "preserving structure, tables, formulas, and reading order.",
                },
                {
                    "from": "gpt",
                    "value": result["text"],
                },
            ],
        }

        if is_train:
            train_samples.append(sample)
        elif is_eval:
            eval_samples.append(sample)
        else:
            train_samples.append(sample)

    output_base = Path(lavida_data_root) / "olmocr"
    output_base.mkdir(parents=True, exist_ok=True)

    if train_samples:
        train_json = output_base / "olmocr_train.json"
        with open(train_json, "w", encoding="utf-8") as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)
        print(f"✓ Train JSON: {train_json} ({len(train_samples)} samples)")

    if eval_samples:
        eval_json = output_base / "olmocr_eval.json"
        with open(eval_json, "w", encoding="utf-8") as f:
            json.dump(eval_samples, f, indent=2, ensure_ascii=False)
        print(f"✓ Eval JSON: {eval_json} ({len(eval_samples)} samples)")

    return len(train_samples), len(eval_samples)


def main():
    """Main conversion pipeline"""

    # You can override these paths via environment variables if needed
    OLMOCR_BASE = os.getenv(
        "OLMOCR_BASE",
        "/path/to/olmocr-dataset",  # Root of extracted olmOCR training data
    )
    LAVIDA_REPO = os.getenv(
        "LAVIDA_REPO",
        "/path/to/LaViDa",  # LaViDa repository root
    )

    DPI = 150  # high quality

    # Auto-detect workers from Slurm or local machine
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        cpus = int(slurm_cpus)
    else:
        cpus = multiprocessing.cpu_count()

    # Optional hard cap via env, default 96 (good for full standard96 node)
    max_workers = int(os.getenv("MAX_WORKERS", "96"))
    N_WORKERS = min(cpus, max_workers)

    print(
        f"Detected CPUs: {cpus}, using N_WORKERS = {N_WORKERS} "
        f"(SLURM_CPUS_PER_TASK={slurm_cpus}, MAX_WORKERS={max_workers})"
    )

    print("=" * 70)
    print("olmOCR → LaViDa Parallel Converter (OPTIMIZED - HIGH QUALITY)")
    print("=" * 70)
    print(f"Source:  {OLMOCR_BASE}")
    print(f"LaViDa:  {LAVIDA_REPO}")
    print(f"DPI:     {DPI}")
    print(f"Workers: {N_WORKERS}")
    print("=" * 70 + "\n")

    lavida_path = Path(LAVIDA_REPO)
    olmocr_path = Path(OLMOCR_BASE)

    lavida_data = lavida_path / "data"
    images_dir = lavida_data / "olmocr" / "images"

    start_time = time.time()
    pairs = find_all_pdf_md_pairs(olmocr_path)

    if not pairs:
        print("✗ No PDF-MD pairs found! Check your olmOCR path.")
        return

    results = parallel_convert_all(pairs, images_dir, dpi=DPI, n_workers=N_WORKERS)

    if not results:
        print("✗ No successful conversions!")
        return

    print("\n" + "=" * 70)
    print("Creating LaViDa format JSON files...")
    print("=" * 70)

    n_train, n_eval = create_lavida_json(results, lavida_data)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"Total samples: {len(results)}")
    print(f"  Train: {n_train}")
    print(f"  Eval:  {n_eval}")
    print(f"Time:   {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Speed:  {len(results)/elapsed:.2f} samples/sec")
    print(f"\nData location: {lavida_data / 'olmocr'}")
    print("Original PDFs: Preserved ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
