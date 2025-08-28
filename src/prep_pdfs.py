from __future__ import annotations

"""Dataset preparation: validate PDF filenames, check duplicates, and report.

Scans ``data/pdfs/`` for ``.pdf`` files, computes file sizes and SHA256 hashes,
flags duplicate contents and invalid filenames, and writes a CSV report to
``reports/pdfs_report.csv`` plus a human-readable summary at
``reports/pdfs_report.txt``.

This phase does NOT move files; the labeling setup script performs staging.
"""

from pathlib import Path
import csv
import hashlib
import re
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"
REPORTS_DIR = ROOT / "reports"
CSV_PATH = REPORTS_DIR / "pdfs_report.csv"
TXT_PATH = REPORTS_DIR / "pdfs_report.txt"


VALID_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.pdf$")
MAX_NAME_LEN = 128


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = [p for p in sorted(PDF_DIR.glob("**/*.pdf")) if p.is_file()]
    seen_hash_to_first: dict[str, Path] = {}

    rows: list[dict[str, str | int | bool]] = []
    for p in pdfs:
        size = p.stat().st_size
        digest = sha256sum(p)

        is_duplicate = digest in seen_hash_to_first
        duplicate_of = seen_hash_to_first.get(digest)
        if not is_duplicate:
            seen_hash_to_first[digest] = p

        name_ok = bool(VALID_NAME_RE.match(p.name)) and len(p.name) <= MAX_NAME_LEN
        reason = ""
        if not name_ok:
            if not VALID_NAME_RE.match(p.name):
                reason = "invalid_chars_or_extension"
            elif len(p.name) > MAX_NAME_LEN:
                reason = "name_too_long"

        rows.append(
            {
                "filename": str(p.relative_to(PDF_DIR)),
                "size_bytes": size,
                "sha256": digest,
                "is_duplicate": is_duplicate,
                "duplicate_of": str(duplicate_of.relative_to(PDF_DIR)) if duplicate_of else "",
                "is_valid_name": name_ok,
                "invalid_reason": reason,
            }
        )

    # Write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "size_bytes",
                "sha256",
                "is_duplicate",
                "duplicate_of",
                "is_valid_name",
                "invalid_reason",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write TXT summary
    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write(f"PDF Preparation Report {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Scanned directory: {PDF_DIR}\n")
        f.write(f"Total PDFs: {len(rows)}\n")
        dupes = sum(1 for r in rows if r["is_duplicate"])  # type: ignore[arg-type]
        invalids = [r for r in rows if not r["is_valid_name"]]  # type: ignore[index]
        f.write(f"Duplicates: {dupes}\n")
        f.write(f"Invalid names: {len(invalids)}\n")
        if invalids:
            f.write("Invalid examples:\n")
            for r in invalids[:10]:
                f.write(f" - {r['filename']}: {r['invalid_reason']}\n")


if __name__ == "__main__":
    main()


