from __future__ import annotations

import io
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF
from PIL import Image
import yaml
from tqdm import tqdm

from utils import ensure_dir


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
PDF_DIR = ROOT / "data" / "pdfs"
OUT_IMG_DIR = ROOT / "data" / "raw" / "images"
LOG_PATH = ROOT / "reports" / "convert_log.txt"


def load_params(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_resize(resize_str: str | None) -> tuple[int, int] | None:
    if not resize_str:
        return None
    try:
        w_str, h_str = str(resize_str).lower().split("x")
        return int(w_str), int(h_str)
    except Exception as exc:
        raise ValueError(
            f"Invalid resize specification: {resize_str!r}. Use 'WIDTHxHEIGHT' or leave empty."
        ) from exc


def convert_pdf(pdf_path: Path, dpi: int, color_mode: str, resize_to: tuple[int, int] | None, fmt: str) -> list[Path]:
    outputs: list[Path] = []
    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=dpi)
            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))

            if color_mode.upper() in {"RGB", "L"}:
                pil_img = pil_img.convert(color_mode.upper())
            else:
                raise ValueError(f"Unsupported color_mode: {color_mode}")

            if resize_to:
                pil_img = pil_img.resize(resize_to, Image.LANCZOS)

            out_name = f"{pdf_path.stem}_p{page_index + 1}.{fmt.lower()}"
            out_path = OUT_IMG_DIR / out_name
            pil_img.save(out_path, format=fmt.upper())
            outputs.append(out_path)
    finally:
        doc.close()
    return outputs


def main() -> None:
    ensure_dir(OUT_IMG_DIR)
    ensure_dir(LOG_PATH.parent)

    params = load_params(PARAMS_PATH)
    conv = params.get("conversion", {})
    dpi = int(conv.get("dpi", 300))
    fmt = str(conv.get("format", "png")).lower()
    color_mode = str(conv.get("color_mode", "RGB"))
    resize_to = parse_resize(conv.get("resize"))

    best_practices = (
        "300 DPI for fine details; PNG to avoid artifacts; grayscale for non-color tasks;"
        " convert per page for scalability; validate outputs after conversion."
    )

    converted_total = 0
    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        logf.write(f"\n=== Conversion run {datetime.utcnow().isoformat()}Z ===\n")
        logf.write(f"Params: dpi={dpi}, format={fmt}, color_mode={color_mode}, resize={resize_to}\n")
        logf.write(f"Notes: {best_practices}\n")

        pdf_files = sorted([p for p in PDF_DIR.glob("**/*.pdf") if p.is_file()])
        if not pdf_files:
            logf.write("No PDFs found.\n")
            return

        for pdf_path in tqdm(pdf_files, desc="Converting PDFs", unit="pdf"):
            try:
                outputs = convert_pdf(pdf_path, dpi=dpi, color_mode=color_mode, resize_to=resize_to, fmt=fmt)
                converted_total += len(outputs)
                for op in outputs:
                    logf.write(f"OK {pdf_path.name} -> {op.relative_to(ROOT)}\n")
            except Exception as exc:
                logf.write(f"ERR {pdf_path.name}: {exc}\n")

        logf.write(f"Total images written: {converted_total}\n")


if __name__ == "__main__":
    main()


