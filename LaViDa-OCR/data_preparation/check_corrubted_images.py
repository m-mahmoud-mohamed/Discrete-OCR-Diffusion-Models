from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm

# Root of your LaViDa dataset repo
ROOT = Path("/path/to/LaViDa")  # LaViDa repository root

# Input / output paths
JSON_IN = ROOT / "data/olmocr/olmocr_train.json"
JSON_OUT = ROOT / "data/olmocr/olmocr_train.cleaned.json"
BAD_LIST = ROOT / "data/olmocr/olmocr_train_bad_images.txt"

def is_image_ok(path: Path) -> bool:
    """Return True if image can be opened and verified, else False."""
    try:
        with Image.open(path) as img:
            img.verify()  # checks internal consistency without decoding full image [web:29][web:31]
        return True
    except Exception:
        return False

def main():
    print(f"Loading JSON: {JSON_IN}")
    with JSON_IN.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total samples in JSON: {len(data)}")

    clean = []
    bad = []

    for sample in tqdm(data, desc="Checking images"):
        img_rel = sample["image"]                 # e.g. "data/olmocr/images/xxx.png"
        img_path = ROOT / img_rel

        if is_image_ok(img_path):
            clean.append(sample)
        else:
            bad.append(img_rel)

    print(f"\nDone.")
    print(f"Good samples : {len(clean)}")
    print(f"Bad images   : {len(bad)}")

    # Save list of bad image paths (relative to ROOT)
    if bad:
        BAD_LIST.parent.mkdir(parents=True, exist_ok=True)
        with BAD_LIST.open("w", encoding="utf-8") as f:
            f.write("\n".join(bad))
        print(f"Bad image list written to: {BAD_LIST}")

    # Save cleaned JSON
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with JSON_OUT.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"Cleaned JSON written to: {JSON_OUT}")

if __name__ == "__main__":
    main()
