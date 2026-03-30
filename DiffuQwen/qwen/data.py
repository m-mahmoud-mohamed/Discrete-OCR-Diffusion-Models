"""
Dataset for DiffuQwen-VL OCR Training.

Loads PDF-Markdown pairs from olmOCR-mix-1025 dataset:
- Recursively walks directories to find PDF-MD pairs by basename
- Converts PDFs to images using pdftoppm (same as olmOCR)
- Preprocesses images to match Qwen2.5-VL requirements
- Returns (image, markdown_text) pairs for training
"""

import io
import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)

# Default paths
DATASET_ROOT = "/path/to/olmocr-dataset"

# olmOCR-style prompt (from build_no_anchoring_v4_yaml_prompt without YAML footer)
DEFAULT_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LaTeX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)"
)

# Image preprocessing constants (same as olmOCR v0.4.0 config)
TARGET_LONGEST_IMAGE_DIM = 1288  # olmOCR default from qwen25_vl_olmocrv4_rotation_1epoch_mix_1025.yaml


def find_pdf_md_pairs(
    root_dir: str,
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Recursively find PDF-Markdown pairs in a directory.
    
    Pairs are matched by basename (same name, different extension).
    
    Args:
        root_dir: Root directory to search
        max_samples: Maximum number of pairs to return (None for all)
    
    Returns:
        List of (pdf_path, md_path) tuples
    """
    pairs = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        logger.warning(f"Directory does not exist: {root_dir}")
        return pairs
    
    # Find all markdown files
    md_files = {}
    for md_path in root_path.rglob("*.md"):
        basename = md_path.stem
        md_files[str(md_path.parent / basename)] = str(md_path)
    
    # Find matching PDFs
    for pdf_path in root_path.rglob("*.pdf"):
        basename = pdf_path.stem
        key = str(pdf_path.parent / basename)
        
        if key in md_files:
            pairs.append((str(pdf_path), md_files[key]))
            
            if max_samples is not None and len(pairs) >= max_samples:
                break
    
    logger.info(f"Found {len(pairs)} PDF-MD pairs in {root_dir}")
    return pairs


def find_image_md_pairs(
    root_dir: str,
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Recursively find image-Markdown pairs in a directory.
    
    Alternative to PDF pairs for pre-rendered images.
    
    Args:
        root_dir: Root directory to search
        image_extensions: Allowed image extensions
        max_samples: Maximum number of pairs to return
    
    Returns:
        List of (image_path, md_path) tuples
    """
    pairs = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        logger.warning(f"Directory does not exist: {root_dir}")
        return pairs
    
    # Find all markdown files
    md_files = {}
    for md_path in root_path.rglob("*.md"):
        basename = md_path.stem
        md_files[str(md_path.parent / basename)] = str(md_path)
    
    # Find matching images
    for ext in image_extensions:
        for img_path in root_path.rglob(f"*{ext}"):
            basename = img_path.stem
            # Remove page suffix if present (e.g., "doc_page1" -> "doc")
            base_key = re.sub(r"_page\d+$", "", basename)
            key = str(img_path.parent / base_key)
            
            # Also try exact match
            exact_key = str(img_path.parent / basename)
            
            if exact_key in md_files:
                pairs.append((str(img_path), md_files[exact_key]))
            elif key in md_files:
                pairs.append((str(img_path), md_files[key]))
            
            if max_samples is not None and len(pairs) >= max_samples:
                return pairs
    
    logger.info(f"Found {len(pairs)} image-MD pairs in {root_dir}")
    return pairs


def get_pdf_media_box_width_height(local_pdf_path: str, page_num: int) -> Tuple[float, float]:
    """
    Get the MediaBox dimensions for a specific page in a PDF file using pdfinfo.
    
    Args:
        local_pdf_path: Path to the PDF file
        page_num: The page number (1-indexed)
    
    Returns:
        Tuple of (width, height) in points
    """
    command = ["pdfinfo", "-f", str(page_num), "-l", str(page_num), "-box", "-enc", "UTF-8", local_pdf_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")
    
    for line in result.stdout.splitlines():
        if "MediaBox" in line:
            media_box_str: List[str] = line.split(":")[1].strip().split()
            media_box: List[float] = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])
    
    raise ValueError("MediaBox not found in the PDF info.")


def load_pdf_as_image(
    pdf_path: str,
    page: int = 0,
    target_longest_image_dim: int = TARGET_LONGEST_IMAGE_DIM,
) -> Optional[Image.Image]:
    """
    Load a PDF page as a PIL Image using pdftoppm (same as olmOCR).
    
    Args:
        pdf_path: Path to PDF file
        page: Page number to load (0-indexed)
        target_longest_image_dim: Target size for longest dimension (default 2048)
    
    Returns:
        PIL Image or None if loading fails
    """
    page_num = page + 1  # pdftoppm is 1-indexed
    
    try:
        # Get PDF page dimensions
        longest_dim = max(get_pdf_media_box_width_height(pdf_path, page_num))
        
        # Calculate DPI to achieve target size (72 points per inch)
        dpi = target_longest_image_dim * 72 / longest_dim
        
        # Convert PDF page to PNG using pdftoppm
        result = subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-f", str(page_num),
                "-l", str(page_num),
                "-r", str(dpi),
                pdf_path,
            ],
            timeout=120,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        if result.returncode != 0:
            logger.error(f"pdftoppm failed for {pdf_path}: {result.stderr.decode()}")
            return None
        
        # Load PNG from stdout
        image = Image.open(io.BytesIO(result.stdout))
        return image.convert("RGB")
        
    except Exception as e:
        logger.error(f"Failed to load PDF {pdf_path}: {e}")
        return None


def resize_image(
    image: Image.Image,
    max_size: int = TARGET_LONGEST_IMAGE_DIM,
) -> Image.Image:
    """
    Resize image to have longest dimension <= max_size.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def strip_yaml_front_matter(content: str) -> str:
    """
    Strip YAML front matter from markdown content.
    
    Front matter format:
    ---
    primary_language: en
    is_rotation_valid: True
    ...
    ---
    
    Args:
        content: Raw markdown content
    
    Returns:
        Content with front matter removed
    """
    # Match YAML front matter at the start of the file
    # Pattern: starts with ---, followed by content, followed by ---
    pattern = r'^---\s*\n.*?\n---\s*\n?'
    cleaned = re.sub(pattern, '', content, count=1, flags=re.DOTALL)
    return cleaned.strip()


def load_markdown(md_path: str, strip_front_matter: bool = True) -> str:
    """
    Load markdown content from file.
    
    Args:
        md_path: Path to markdown file
        strip_front_matter: Whether to strip YAML front matter (default True)
    
    Returns:
        Markdown content as string
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    if strip_front_matter:
        content = strip_yaml_front_matter(content)
    
    return content


class OLMoCRDataset(Dataset):
    """
    Dataset for loading olmOCR PDF-to-Markdown pairs.
    
    Supports:
    - PDF files (converted to images)
    - Pre-rendered images
    - Recursive directory traversal
    - Image preprocessing for Qwen2.5-VL
    """
    
    def __init__(
        self,
        root_dir: str = DATASET_ROOT,
        split: str = "train",
        prompt: str = DEFAULT_PROMPT,
        max_samples: Optional[int] = None,
        target_image_size: int = TARGET_LONGEST_IMAGE_DIM,
        use_images: bool = False,
        subdirs: Optional[List[str]] = None,
        processor: Optional[Any] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory of olmOCR dataset
            split: "train" or "eval"
            prompt: Prompt text to prepend
            max_samples: Maximum samples to load (None for all)
            target_image_size: Target longest image dimension
            use_images: If True, look for image files instead of PDFs
            subdirs: Specific subdirectories to search (e.g., ["processed_00_documents"])
            processor: Optional Qwen processor for preprocessing
        """
        self.root_dir = root_dir
        self.split = split
        self.prompt = prompt
        self.target_image_size = target_image_size
        self.use_images = use_images
        self.processor = processor
        
        # Determine directories to search
        if subdirs is not None:
            search_dirs = [
                os.path.join(root_dir, f"{d}_{split}")
                for d in subdirs
            ]
        else:
            # Search all split directories
            search_dirs = [
                os.path.join(root_dir, d)
                for d in os.listdir(root_dir)
                if d.endswith(f"_{split}") and os.path.isdir(os.path.join(root_dir, d))
            ]
        
        # Find all pairs
        self.pairs = []
        remaining_samples = max_samples
        
        for search_dir in search_dirs:
            if use_images:
                found_pairs = find_image_md_pairs(search_dir, max_samples=remaining_samples)
            else:
                found_pairs = find_pdf_md_pairs(search_dir, max_samples=remaining_samples)
            
            self.pairs.extend(found_pairs)
            
            if max_samples is not None:
                remaining_samples = max_samples - len(self.pairs)
                if remaining_samples <= 0:
                    break
        
        logger.info(f"Initialized {split} dataset with {len(self.pairs)} samples")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - "image": PIL Image
            - "text": Markdown ground truth
            - "prompt": Input prompt
            - "source_path": Path to source file
        """
        source_path, md_path = self.pairs[idx]
        
        # Load image
        if source_path.endswith(".pdf"):
            image = load_pdf_as_image(source_path, page=0, target_longest_image_dim=self.target_image_size)
            if image is None:
                # Return a placeholder for failed loads
                logger.warning(f"Failed to load {source_path}, using placeholder")
                image = Image.new("RGB", (224, 224), color="white")
        else:
            image = Image.open(source_path).convert("RGB")
            # Resize image if needed
            image = resize_image(image, self.target_image_size)
        
        # Load markdown
        text = load_markdown(md_path)
        
        return {
            "image": image,
            "text": text,
            "prompt": self.prompt,
            "source_path": source_path,
        }
    
    def get_subset(self, indices: List[int]) -> "OLMoCRDataset":
        """
        Create a subset of the dataset.
        
        Args:
            indices: List of indices to include
        
        Returns:
            New dataset with only specified indices
        """
        subset = OLMoCRDataset.__new__(OLMoCRDataset)
        subset.root_dir = self.root_dir
        subset.split = self.split
        subset.prompt = self.prompt
        subset.target_image_size = self.target_image_size
        subset.use_images = self.use_images
        subset.processor = self.processor
        subset.pairs = [self.pairs[i] for i in indices]
        return subset


class OLMoCRDatasetFromList(Dataset):
    """
    Dataset from explicit list of (image_path, md_path) pairs.
    
    Useful for custom splits or filtered datasets.
    """
    
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        prompt: str = DEFAULT_PROMPT,
        target_image_size: int = TARGET_LONGEST_IMAGE_DIM,
    ):
        """
        Initialize from explicit list of pairs.
        
        Args:
            pairs: List of (image/pdf_path, md_path) tuples
            prompt: Prompt text
            target_image_size: Target longest image dimension
        """
        self.pairs = pairs
        self.prompt = prompt
        self.target_image_size = target_image_size
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        source_path, md_path = self.pairs[idx]
        
        # Load image
        if source_path.endswith(".pdf"):
            image = load_pdf_as_image(source_path, page=0, target_longest_image_dim=self.target_image_size)
            if image is None:
                image = Image.new("RGB", (224, 224), color="white")
        else:
            image = Image.open(source_path).convert("RGB")
            # Resize if needed
            image = resize_image(image, self.target_image_size)
        
        # Load markdown
        text = load_markdown(md_path)
        
        return {
            "image": image,
            "text": text,
            "prompt": self.prompt,
            "source_path": source_path,
        }


def create_train_eval_split(
    root_dir: str = DATASET_ROOT,
    train_subdirs: Optional[List[str]] = None,
    eval_subdirs: Optional[List[str]] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    **kwargs,
) -> Tuple[OLMoCRDataset, OLMoCRDataset]:
    """
    Create train and eval datasets.
    
    Args:
        root_dir: Root directory of olmOCR dataset
        train_subdirs: Subdirectories for training (e.g., ["processed_00_documents"])
        eval_subdirs: Subdirectories for evaluation
        max_train_samples: Maximum training samples
        max_eval_samples: Maximum evaluation samples
        **kwargs: Additional arguments for OLMoCRDataset
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    train_dataset = OLMoCRDataset(
        root_dir=root_dir,
        split="train",
        subdirs=train_subdirs,
        max_samples=max_train_samples,
        **kwargs,
    )
    
    eval_dataset = OLMoCRDataset(
        root_dir=root_dir,
        split="eval",
        subdirs=eval_subdirs,
        max_samples=max_eval_samples,
        **kwargs,
    )
    
    return train_dataset, eval_dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Optional[callable] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        collate_fn: Custom collation function
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        **kwargs,
    )
