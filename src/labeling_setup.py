from __future__ import annotations

"""Labeling setup orchestrator with a class-based API.

Prepares staging for either YOLO or COCO labeling conventions, copying PDFs
and rendering page images to staging.
"""

from pathlib import Path
from typing import Literal
import argparse
import shutil

import pypdfium2 as pdfium
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"
STAGING_DIR = ROOT / "data" / "staging"


class LabelingSetup:
    def __init__(self, convention: Literal["yolo", "coco"]) -> None:
        self.convention = convention
        self.staging_conv_dir = STAGING_DIR / convention
        (self.staging_conv_dir / "images").mkdir(parents=True, exist_ok=True)
        if convention == "yolo":
            (self.staging_conv_dir / "labels").mkdir(parents=True, exist_ok=True)
        else:
            (self.staging_conv_dir / "annotations").mkdir(parents=True, exist_ok=True)
        self.staging_pdfs_dir = STAGING_DIR / "pdfs"
        self.staging_pdfs_dir.mkdir(parents=True, exist_ok=True)

    def render_pdf_to_dir(self, pdf_path: Path, out_images_dir: Path, dpi: int = 300, color_mode: str = "RGB") -> None:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            scale = max(float(dpi) / 72.0, 0.1)
            for page_index in range(len(pdf)):
                page = pdf.get_page(page_index)
                try:
                    bitmap = page.render(scale=scale)
                    pil_img = bitmap.to_pil().convert(color_mode.upper())
                    out_name = f"{pdf_path.stem}_p{page_index + 1}.png"
                    pil_img.save(out_images_dir / out_name, format="PNG")
                finally:
                    page.close()
        finally:
            pdf.close()

    def copy_and_render(self) -> None:
        # Optionally read prep report to filter out duplicates/invalids
        valid_set: set[str] = set()
        report_csv = ROOT / "reports" / "pdfs_report.csv"
        if report_csv.exists():
            import csv

            with open(report_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    is_dup = str(row.get("is_duplicate", "false")).lower() in {"true", "1", "yes"}
                    is_valid = str(row.get("is_valid_name", "true")).lower() in {"true", "1", "yes"}
                    if (not is_dup) and is_valid:
                        valid_set.add(str(row["filename"]))

        src_pdfs = [p for p in sorted(PDF_DIR.glob("**/*.pdf")) if p.is_file()]
        for pdf in src_pdfs:
            rel = str(pdf.relative_to(PDF_DIR))
            if valid_set and rel not in valid_set:
                continue
            dst = self.staging_pdfs_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, dst)
            self.render_pdf_to_dir(dst, self.staging_conv_dir / "images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare staging directory for labeling")
    parser.add_argument("--format", dest="fmt", default=None, help="Annotation format: yolo or coco")
    args = parser.parse_args()
    fmt = (args.fmt or "yolo").strip().lower()
    if fmt not in {"yolo", "coco"}:
        raise SystemExit("--format must be 'yolo' or 'coco'")
    setup = LabelingSetup(fmt)  # type: ignore[arg-type]
    setup.copy_and_render()
    print(f"Staging prepared under: {setup.staging_conv_dir}")


if __name__ == "__main__":
    main()


