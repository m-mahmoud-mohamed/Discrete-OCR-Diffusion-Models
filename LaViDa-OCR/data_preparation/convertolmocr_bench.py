from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm

base_pdf_dir = Path("/path/to/olmOCR-bench/bench_data/pdfs")  # olmOCR-bench PDF directory
base_img_dir = Path("/path/to/bench_images")  # Output directory for rendered images

categories = ["arxiv_math", "headers_footers", "long_tiny_text", "multi_column", "old_scans", "old_scans_math", "tables"]
total_pdfs = 0
converted_pdfs = 0
failed_pdfs = []

for category in tqdm(categories, desc="Categories"):
    pdf_dir = base_pdf_dir / category
    img_dir = base_img_dir / category
    img_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    for pdf_path in tqdm(pdf_files, desc=f"{category}", leave=False):
        total_pdfs += 1
        try:
            pages = convert_from_path(str(pdf_path), dpi=150)
            if not pages:
                raise Exception("PDF contains zero pages")
            if len(pages) == 1:
                img_filename = f"{pdf_path.stem}.png"
                pages[0].save(img_dir / img_filename, "PNG")
            else:
                for i, page in enumerate(pages):
                    img_filename = f"{pdf_path.stem}_p{i+1}.png"
                    page.save(img_dir / img_filename, "PNG")
            converted_pdfs += 1
        except Exception as e:
            tqdm.write(f"Failed to convert {pdf_path}: {e}")
            failed_pdfs.append(str(pdf_path))

print(f"\nTotal PDFs found: {total_pdfs}")
print(f"PDFs converted: {converted_pdfs}")
print(f"PDFs failed: {len(failed_pdfs)}")
if failed_pdfs:
    print("Failed files:")
    for f in failed_pdfs:
        print(f)
