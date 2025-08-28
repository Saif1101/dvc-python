from __future__ import annotations

"""Create or extend a versioned dataset from staged labeled data.

Supports YOLO staging produced by ``setup_labeling.py``.

New datasets are created under ``data/versions/vNNN`` (e.g., ``v001``). If
extending an existing version, a derived name like ``v001-suffix`` is used.

The script splits into ``train/valid/test`` using a 70/20/10 ratio and updates
``data/data.yaml`` paths to point to the versioned dataset.
"""

from pathlib import Path
import argparse
import re
from typing import Dict, Tuple

from utils import split_dataset


ROOT = Path(__file__).resolve().parents[1]
VERSIONS_DIR = ROOT / "data" / "versions"
STAGING_YOLO = ROOT / "data" / "staging" / "yolo"
DATA_YAML = ROOT / "data" / "data.yaml"


class DatasetVersioner:
    def __init__(self, versions_dir: Path, data_yaml: Path) -> None:
        self.versions_dir = versions_dir
        self.data_yaml = data_yaml

    def next_version_name(self) -> str:
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        existing = [p.name for p in self.versions_dir.glob("v*") if p.is_dir()]
        nums = []
        for name in existing:
            m = re.match(r"^v(\d{3})(?:[\w-].*)?$", name)
            if m:
                nums.append(int(m.group(1)))
        n = max(nums) + 1 if nums else 1
        return f"v{n:03d}"


def prompt_choice(prompt: str, options: Dict[str, str]) -> str:
    print(prompt)
    for k, v in options.items():
        print(f"  [{k}] {v}")
    while True:
        ans = input("> ").strip().lower()
        if ans in options:
            return ans
        print("Please choose one of:", ", ".join(options))


    def update_data_yaml_to_version(self, version_dir: Path) -> None:
        train = version_dir / "train" / "images"
        val = version_dir / "valid" / "images"
        test = version_dir / "test" / "images"
        lines = [
            f"path: {version_dir.as_posix()}",
            f"train: {train.as_posix()}",
            f"val: {val.as_posix()}",
            f"test: {test.as_posix()}",
            "",
        ]
        try:
            existing = self.data_yaml.read_text(encoding="utf-8").splitlines()
            for ln in existing:
                if ln.strip().startswith("names:") or ln.strip().startswith("nc:"):
                    lines.append(ln)
        except Exception:
            pass
        self.data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or extend a versioned dataset from staging")
    parser.add_argument("--mode", choices=["new", "extend"], default=None, help="Create new or extend existing")
    parser.add_argument("--suffix", default=None, help="Suffix for derived dataset when extending (e.g., 'aug1')")
    parser.add_argument("--splits", default="0.7,0.2,0.1", help="Train,valid,test split (sum to 1.0)")
    args = parser.parse_args()

    if not STAGING_YOLO.exists():
        raise SystemExit(f"Expected YOLO staging at {STAGING_YOLO} (run setup_labeling.py in 'yolo' mode)")

    try:
        parts = [float(x) for x in args.splits.split(",")]
        assert len(parts) == 3
        assert abs(sum(parts) - 1.0) < 1e-6
        splits = (parts[0], parts[1], parts[2])  # type: ignore[assignment]
    except Exception:
        raise SystemExit("Invalid --splits. Use e.g. 0.7,0.2,0.1")

    mode = args.mode or prompt_choice("Dataset versioning mode:", {"n": "new", "e": "extend"})
    versioner = DatasetVersioner(VERSIONS_DIR, DATA_YAML)
    if mode in {"n", "new"}:
        version_name = versioner.next_version_name()
    else:
        # choose existing base
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        existing = sorted([p for p in VERSIONS_DIR.glob("v*") if p.is_dir()])
        if not existing:
            version_name = versioner.next_version_name()
        else:
            print("Choose base version to extend:")
            for idx, p in enumerate(existing, start=1):
                print(f"  [{idx}] {p.name}")
            while True:
                sel = input("> ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(existing):
                    base = existing[int(sel) - 1]
                    break
                print("Enter a number from the list.")
            suffix = args.suffix or input("Suffix for derived dataset (e.g., 'aug1'): ").strip()
            version_name = f"{base.name}-{suffix}" if suffix else f"{base.name}-derived"

    version_dir = VERSIONS_DIR / version_name
    (version_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (version_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (version_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (version_dir / "valid" / "labels").mkdir(parents=True, exist_ok=True)
    (version_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
    (version_dir / "test" / "labels").mkdir(parents=True, exist_ok=True)

    # Perform split/copy from staging
    split_dataset(
        images_dir=STAGING_YOLO / "images",
        labels_dir=STAGING_YOLO / "labels",
        out_dir=version_dir,
        splits=splits,  # type: ignore[arg-type]
    )

    versioner.update_data_yaml_to_version(version_dir)
    print(f"Dataset version ready: {version_dir}")
    print(f"Updated {DATA_YAML} to point to this version.")


if __name__ == "__main__":
    main()


