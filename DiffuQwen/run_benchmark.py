#!/usr/bin/env python3
"""
Benchmark script for running DiffuQwen on olmOCR-bench dataset.

Usage:
    python run_benchmark.py --checkpoint /path/to/checkpoint --output_dir ./benchmark_results
    
    # With specific category
    python run_benchmark.py --checkpoint /path/to/checkpoint --category arxiv_math
    
    # With KV cache for faster inference
    python run_benchmark.py --checkpoint /path/to/checkpoint --use_cache
    
    # Compare AR vs Diffusion
    python run_benchmark.py --checkpoint /path/to/checkpoint --no_diffusion --output_dir ./ar_results
    
    # Multi-GPU (automatically uses all available GPUs)
    python run_benchmark.py --checkpoint /path/to/checkpoint --num_gpus 4
    
    # Specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_benchmark.py --checkpoint /path/to/checkpoint --num_gpus 4
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from infer import (
    MASK_TOKEN_ID,
    DEFAULT_PROMPT,
    load_model,
    generate,
)


# Default paths
DEFAULT_BENCH_DATA_PATH = "/path/to/olmOCR-bench/bench_data/pdfs"
DEFAULT_BASE_MODEL = "/path/to/olmocr-model"

# Categories in the benchmark
CATEGORIES = [
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "tables",
]


def get_pdf_image(pdf_path: Path) -> Optional[Image.Image]:
    """Render PDF to image using olmOCR's pdf rendering."""
    try:
        from olmocr.data.renderpdf import render_pdf_to_base64png
        import base64
        from io import BytesIO
        
        base64_png = render_pdf_to_base64png(str(pdf_path), page_num=0, target_longest_image_dim=1024)
        image_data = base64.b64decode(base64_png)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error rendering PDF {pdf_path}: {e}")
        return None


def run_benchmark(
    model,
    processor,
    tokenizer,
    bench_data_path: Path,
    output_dir: Path,
    categories: Optional[List[str]] = None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 2048,
    num_steps: int = 64,
    temperature: float = 0.5,
    top_p: float = 0.95,
    top_k: int = 50,
    cfg_weight: float = 1.5,
    use_diffusion: bool = True,
    use_cache: bool = False,
    device: str = "cuda",
    max_samples_per_category: Optional[int] = None,
    skip_existing: bool = False,
):
    """Run benchmark on olmOCR-bench data."""
    
    if categories is None:
        categories = CATEGORIES
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {
        "total_pdfs": 0,
        "successful": 0,
        "failed": 0,
        "total_time": 0,
        "per_category": {},
    }
    
    for category in categories:
        category_path = bench_data_path / category
        if not category_path.exists():
            print(f"Category {category} not found at {category_path}, skipping...")
            continue
        
        # Get all PDFs in category
        pdf_files = sorted(category_path.glob("*.pdf"))
        if max_samples_per_category:
            pdf_files = pdf_files[:max_samples_per_category]
        
        if not pdf_files:
            print(f"No PDFs found in {category}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing category: {category} ({len(pdf_files)} PDFs)")
        print(f"{'='*60}")
        
        # Create category output directory
        category_output_dir = output_dir / category
        category_output_dir.mkdir(parents=True, exist_ok=True)
        
        category_stats = {
            "total": len(pdf_files),
            "successful": 0,
            "failed": 0,
            "times": [],
        }
        
        for pdf_file in tqdm(pdf_files, desc=category):
            stats["total_pdfs"] += 1
            
            # Check if output already exists
            output_file = category_output_dir / f"{pdf_file.stem}.md"
            if skip_existing and output_file.exists():
                category_stats["successful"] += 1
                stats["successful"] += 1
                continue
            
            # Render PDF to image
            image = get_pdf_image(pdf_file)
            if image is None:
                category_stats["failed"] += 1
                stats["failed"] += 1
                continue
            
            # Generate output
            start_time = time.time()
            try:
                # generate() returns (outputs, inference_times, intermediates)
                result = generate(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    images=[image],
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    num_steps=num_steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    cfg_weight=cfg_weight,
                    use_diffusion=use_diffusion,
                    use_cache=use_cache,
                    device=device,
                )
                # Unpack the tuple: (list_of_outputs, inference_times, intermediates)
                generated_texts, _, _ = result
                output = generated_texts[0]  # First (and only) generated text
                
                # Ensure output is a string
                if not isinstance(output, str):
                    output = str(output)
                
                elapsed = time.time() - start_time
                
                # Save output
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output)
                
                # Save metadata
                meta_file = category_output_dir / f"{pdf_file.stem}_meta.json"
                meta = {
                    "source_pdf": str(pdf_file),
                    "category": category,
                    "generation_time": elapsed,
                    "num_tokens": len(output.split()),  # Approximate
                    "timestamp": datetime.now().isoformat(),
                    "settings": {
                        "max_tokens": max_tokens,
                        "num_steps": num_steps,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "cfg_weight": cfg_weight,
                        "use_diffusion": use_diffusion,
                        "use_cache": use_cache,
                    }
                }
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                
                category_stats["successful"] += 1
                category_stats["times"].append(elapsed)
                stats["successful"] += 1
                stats["total_time"] += elapsed
                
            except Exception as e:
                print(f"\nError processing {pdf_file.name}: {e}")
                category_stats["failed"] += 1
                stats["failed"] += 1
        
        # Calculate category statistics
        if category_stats["times"]:
            category_stats["avg_time"] = sum(category_stats["times"]) / len(category_stats["times"])
            category_stats["min_time"] = min(category_stats["times"])
            category_stats["max_time"] = max(category_stats["times"])
        
        stats["per_category"][category] = category_stats
        
        print(f"\n{category} Results:")
        print(f"  Successful: {category_stats['successful']}/{category_stats['total']}")
        if category_stats["times"]:
            print(f"  Avg time: {category_stats.get('avg_time', 0):.2f}s")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs processed: {stats['total_pdfs']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    if stats["successful"] > 0:
        print(f"Average time per PDF: {stats['total_time'] / stats['successful']:.2f}s")
        print(f"Total time: {stats['total_time']:.2f}s")
    
    # Save overall stats
    stats_file = output_dir / "benchmark_stats.json"
    # Remove non-serializable data
    for cat in stats["per_category"]:
        if "times" in stats["per_category"][cat]:
            del stats["per_category"][cat]["times"]
    
    stats["timestamp"] = datetime.now().isoformat()
    stats["settings"] = {
        "max_tokens": max_tokens,
        "num_steps": num_steps,
        "temperature": temperature,
        "use_diffusion": use_diffusion,
        "use_cache": use_cache,
    }
    
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Statistics saved to: {stats_file}")
    
    return stats


def gpu_worker(
    gpu_id: int,
    pdf_files: List[Path],
    output_dir: Path,
    checkpoint_path: str,
    base_model_path: str,
    prompt: str,
    max_tokens: int,
    num_steps: int,
    temperature: float,
    top_p: float,
    top_k: int,
    cfg_weight: float,
    use_diffusion: bool,
    use_cache: bool,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """
    Worker function that runs on a single GPU.
    
    Args:
        gpu_id: GPU device ID
        pdf_files: List of PDF files to process
        output_dir: Output directory
        ... other generation params
    
    Returns:
        Dict with statistics for this worker
    """
    device = f"cuda:{gpu_id}"
    
    # Set CUDA device for this process
    torch.cuda.set_device(gpu_id)
    
    print(f"[GPU {gpu_id}] Loading model...")
    model, processor, tokenizer = load_model(
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        device=device,
    )
    print(f"[GPU {gpu_id}] Model loaded! Processing {len(pdf_files)} PDFs...")
    
    stats = {
        "gpu_id": gpu_id,
        "total": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "total_time": 0,
        "times": [],
    }
    
    for pdf_file in tqdm(pdf_files, desc=f"GPU {gpu_id}", position=gpu_id):
        # Determine category and output file
        category = pdf_file.parent.name
        category_output_dir = output_dir / category
        category_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = category_output_dir / f"{pdf_file.stem}.md"
        
        # Check if output already exists
        if skip_existing and output_file.exists():
            stats["successful"] += 1
            continue
        
        # Render PDF to image
        image = get_pdf_image(pdf_file)
        if image is None:
            stats["failed"] += 1
            continue
        
        start_time = time.time()
        try:
            result = generate(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                images=[image],
                prompt=prompt,
                max_new_tokens=max_tokens,
                num_steps=num_steps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                cfg_weight=cfg_weight,
                use_diffusion=use_diffusion,
                use_cache=use_cache,
                device=device,
            )
            generated_texts, _, _ = result
            output = generated_texts[0]
            
            if not isinstance(output, str):
                output = str(output)
            
            elapsed = time.time() - start_time
            
            # Save output
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            
            # Save metadata
            meta_file = category_output_dir / f"{pdf_file.stem}_meta.json"
            meta = {
                "source_pdf": str(pdf_file),
                "category": category,
                "generation_time": elapsed,
                "gpu_id": gpu_id,
                "timestamp": datetime.now().isoformat(),
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            
            stats["successful"] += 1
            stats["total_time"] += elapsed
            stats["times"].append(elapsed)
            
        except Exception as e:
            print(f"\n[GPU {gpu_id}] Error processing {pdf_file.name}: {e}")
            stats["failed"] += 1
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return stats


def run_benchmark_multi_gpu(
    bench_data_path: Path,
    output_dir: Path,
    checkpoint_path: str,
    base_model_path: str,
    num_gpus: int,
    categories: Optional[List[str]] = None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 2048,
    num_steps: int = 64,
    temperature: float = 0.5,
    top_p: float = 0.95,
    top_k: int = 50,
    cfg_weight: float = 1.5,
    use_diffusion: bool = True,
    use_cache: bool = False,
    max_samples_per_category: Optional[int] = None,
    skip_existing: bool = False,
):
    """Run benchmark on multiple GPUs."""
    
    if categories is None:
        categories = CATEGORIES
    
    # Collect all PDF files
    all_pdfs = []
    for category in categories:
        category_path = bench_data_path / category
        if not category_path.exists():
            print(f"Category {category} not found at {category_path}, skipping...")
            continue
        
        pdf_files = sorted(category_path.glob("*.pdf"))
        if max_samples_per_category:
            pdf_files = pdf_files[:max_samples_per_category]
        all_pdfs.extend(pdf_files)
    
    if not all_pdfs:
        print("No PDFs found!")
        return
    
    print(f"\nTotal PDFs to process: {len(all_pdfs)}")
    print(f"Using {num_gpus} GPUs")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split PDFs across GPUs
    pdfs_per_gpu = [[] for _ in range(num_gpus)]
    for i, pdf in enumerate(all_pdfs):
        pdfs_per_gpu[i % num_gpus].append(pdf)
    
    for i, pdfs in enumerate(pdfs_per_gpu):
        print(f"  GPU {i}: {len(pdfs)} PDFs")
    
    # Use spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Run workers in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id in range(num_gpus):
            future = executor.submit(
                gpu_worker,
                gpu_id=gpu_id,
                pdf_files=pdfs_per_gpu[gpu_id],
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                base_model_path=base_model_path,
                prompt=prompt,
                max_tokens=max_tokens,
                num_steps=num_steps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                cfg_weight=cfg_weight,
                use_diffusion=use_diffusion,
                use_cache=use_cache,
                skip_existing=skip_existing,
            )
            futures.append(future)
        
        # Collect results
        all_stats = []
        for future in as_completed(futures):
            try:
                stats = future.result()
                all_stats.append(stats)
            except Exception as e:
                print(f"Worker failed: {e}")
    
    total_time = time.time() - start_time
    
    # Aggregate statistics
    total_successful = sum(s["successful"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    total_proc_time = sum(s["total_time"] for s in all_stats)
    
    # Print summary
    print(f"\n{'='*60}")
    print("MULTI-GPU BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs: {len(all_pdfs)}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Wall clock time: {total_time:.2f}s")
    print(f"Total GPU time: {total_proc_time:.2f}s")
    if total_successful > 0:
        print(f"Avg time per PDF: {total_proc_time / total_successful:.2f}s")
        print(f"Throughput: {total_successful / total_time:.2f} PDFs/sec")
    
    # Per-GPU stats
    print(f"\nPer-GPU Statistics:")
    for stats in sorted(all_stats, key=lambda x: x["gpu_id"]):
        gpu_id = stats["gpu_id"]
        print(f"  GPU {gpu_id}: {stats['successful']}/{stats['total']} successful, "
              f"avg {stats['total_time']/max(1,stats['successful']):.2f}s/PDF")
    
    # Save stats
    summary = {
        "total_pdfs": len(all_pdfs),
        "successful": total_successful,
        "failed": total_failed,
        "wall_clock_time": total_time,
        "total_gpu_time": total_proc_time,
        "num_gpus": num_gpus,
        "throughput_pdfs_per_sec": total_successful / total_time if total_time > 0 else 0,
        "timestamp": datetime.now().isoformat(),
        "per_gpu": [{k: v for k, v in s.items() if k != "times"} for s in all_stats],
        "settings": {
            "max_tokens": max_tokens,
            "num_steps": num_steps,
            "temperature": temperature,
            "use_diffusion": use_diffusion,
            "use_cache": use_cache,
        }
    }
    
    stats_file = output_dir / "benchmark_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DiffuQwen on olmOCR-bench dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Paths
    parser.add_argument(
        "--bench_data",
        type=str,
        default=DEFAULT_BENCH_DATA_PATH,
        help="Path to olmOCR-bench PDFs directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Path to base olmOCR model",
    )
    
    # Category selection
    parser.add_argument(
        "--category",
        type=str,
        nargs="+",
        choices=CATEGORIES,
        help="Specific categories to run (default: all)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per category (for quick testing)",
    )
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Generation prompt")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--num_steps", type=int, default=64, help="Diffusion steps")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--cfg_weight", type=float, default=1.5, help="CFG weight")
    
    # Mode selection
    parser.add_argument(
        "--no_diffusion",
        action="store_true",
        help="Use AR instead of diffusion (for comparison)",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use KV caching for faster inference",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip PDFs that already have output files (for resuming interrupted runs)",
    )
    
    # Device / Multi-GPU
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (single GPU mode)")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (>1 enables multi-GPU mode)",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    bench_data_path = Path(args.bench_data)
    if not bench_data_path.exists():
        print(f"Error: Benchmark data path not found: {bench_data_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    # Add timestamp and mode to output directory
    mode = "ar" if args.no_diffusion else "diffusion"
    cache_str = "_cached" if args.use_cache else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"{mode}{cache_str}_{timestamp}"
    
    print(f"Benchmark Data: {bench_data_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {mode}")
    print(f"KV Cache: {args.use_cache}")
    print(f"GPUs: {args.num_gpus}")
    if args.category:
        print(f"Categories: {args.category}")
    else:
        print(f"Categories: all ({len(CATEGORIES)})")
    if args.max_samples:
        print(f"Max samples per category: {args.max_samples}")
    
    # Check GPU availability
    if args.num_gpus > 1:
        available_gpus = torch.cuda.device_count()
        if args.num_gpus > available_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available.")
            args.num_gpus = available_gpus
        
        if args.num_gpus < 2:
            print("Warning: Multi-GPU requested but only 1 GPU available. Using single-GPU mode.")
    
    # Run benchmark
    if args.num_gpus > 1:
        # Multi-GPU mode
        print(f"\n🚀 Running in MULTI-GPU mode with {args.num_gpus} GPUs...")
        run_benchmark_multi_gpu(
            bench_data_path=bench_data_path,
            output_dir=output_dir,
            checkpoint_path=args.checkpoint,
            base_model_path=args.base_model,
            num_gpus=args.num_gpus,
            categories=args.category,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            cfg_weight=args.cfg_weight,
            use_diffusion=not args.no_diffusion,
            use_cache=args.use_cache,
            max_samples_per_category=args.max_samples,
            skip_existing=args.skip_existing,
        )
    else:
        # Single-GPU mode
        print("\nLoading model...")
        model, processor, tokenizer = load_model(
            checkpoint_path=args.checkpoint,
            base_model_path=args.base_model,
            device=args.device,
        )
        print("Model loaded!")
        
        run_benchmark(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            bench_data_path=bench_data_path,
            output_dir=output_dir,
            categories=args.category,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            cfg_weight=args.cfg_weight,
            use_diffusion=not args.no_diffusion,
            use_cache=args.use_cache,
            device=args.device,
            max_samples_per_category=args.max_samples,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
