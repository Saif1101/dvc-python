import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_params(params_path: Path) -> Dict:
    """Load YAML parameters from ``params.yaml`` file."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    """Create directory ``path`` if it does not exist (including parents)."""
    path.mkdir(parents=True, exist_ok=True)


def parse_resize(resize_str: str | None) -> Tuple[int, int] | None:
    """Parse a ``WIDTHxHEIGHT`` string into integers or return ``None``."""
    if not resize_str:
        return None
    try:
        width_str, height_str = str(resize_str).lower().split("x")
        return int(width_str), int(height_str)
    except Exception:
        raise ValueError(
            f"Invalid resize specification: {resize_str!r}. Use 'WIDTHxHEIGHT' or leave empty."
        )


def list_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path | None]]:
    """List image files and their matching YOLO label file (if present)."""
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    pairs: List[Tuple[Path, Path | None]] = []
    for img in sorted(images_dir.glob("**/*")):
        if img.suffix.lower() in image_exts and img.is_file():
            label = labels_dir / (img.stem + ".txt")
            pairs.append((img, label if label.exists() else None))
    return pairs


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    splits: Tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> Dict[str, Dict[str, Path]]:
    """Split a YOLO dataset into train/valid/test subfolders.

    The function preserves file basenames and copies corresponding labels when
    present. Splits are computed by index, not randomized.
    """
    assert abs(sum(splits) - 1.0) < 1e-6, "Splits must sum to 1.0"

    ensure_dir(out_dir / "train" / "images")
    ensure_dir(out_dir / "train" / "labels")
    ensure_dir(out_dir / "valid" / "images")
    ensure_dir(out_dir / "valid" / "labels")
    ensure_dir(out_dir / "test" / "images")
    ensure_dir(out_dir / "test" / "labels")

    pairs = list_image_label_pairs(images_dir, labels_dir)
    num_total = len(pairs)
    n_train = int(num_total * splits[0])
    n_valid = int(num_total * splits[1])
    # remainder goes to test

    def copy_pair(img: Path, lbl: Path | None, dst_img_dir: Path, dst_lbl_dir: Path) -> None:
        shutil.copy2(img, dst_img_dir / img.name)
        if lbl and lbl.exists():
            shutil.copy2(lbl, dst_lbl_dir / lbl.name)

    for idx, (img, lbl) in enumerate(pairs):
        if idx < n_train:
            copy_pair(img, lbl, out_dir / "train" / "images", out_dir / "train" / "labels")
        elif idx < n_train + n_valid:
            copy_pair(img, lbl, out_dir / "valid" / "images", out_dir / "valid" / "labels")
        else:
            copy_pair(img, lbl, out_dir / "test" / "images", out_dir / "test" / "labels")

    return {
        "train": {"images": out_dir / "train" / "images", "labels": out_dir / "train" / "labels"},
        "valid": {"images": out_dir / "valid" / "images", "labels": out_dir / "valid" / "labels"},
        "test": {"images": out_dir / "test" / "images", "labels": out_dir / "test" / "labels"},
    }


def update_data_yaml(data_yaml_path: Path, class_names: List[str]) -> None:
    """Update YOLO data.yaml with class names and count."""
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["names"] = list(class_names)
    data["nc"] = len(class_names)
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


