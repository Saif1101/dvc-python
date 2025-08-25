from __future__ import annotations

"""Convert YOLO-format annotations to COCO for RFDETR.

Reads class names from ``data/data.yaml`` and, for each split under
``data/raw/{train,valid,test}``, scans images and YOLO label files to produce
``_annotations.coco.json`` in the same split directory, compatible with
RFDETR training.
"""

import json
from pathlib import Path
from typing import Dict, List

import yaml
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "data.yaml"
SPLITS = ("train", "valid", "test")


def load_data_yaml(path: Path) -> Dict:
    """Load YOLO ``data.yaml`` to get class names and roots."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def yolo_box_to_coco(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> List[float]:
    """Convert normalized YOLO (cx, cy, w, h) to COCO bbox [x, y, w, h] (pixels)."""
    x = (cx - w / 2.0) * img_w
    y = (cy - h / 2.0) * img_h
    w_abs = w * img_w
    h_abs = h * img_h
    return [max(0.0, x), max(0.0, y), max(0.0, w_abs), max(0.0, h_abs)]


def convert_split(split_dir: Path, class_names: List[str]) -> Path:
    """Create COCO annotations for one split directory.

    Assumes images in ``images/`` and YOLO label files in ``labels/`` with the
    same basenames.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    out_json = split_dir / "_annotations.coco.json"

    images = []
    annotations = []
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]

    image_id = 1
    ann_id = 1

    image_files = sorted([p for p in images_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    for img_path in tqdm(image_files, desc=f"{split_dir.name}: converting", unit="img"):
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            continue

        images.append(
            {
                "id": image_id,
                "file_name": str(img_path.name),
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            try:
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) != 5 and len(parts) != 6:
                        # YOLOv5 sometimes has 6 values including confidence; ignore extra
                        parts = parts[:5]
                    cls_idx = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                    bbox = yolo_box_to_coco(cx, cy, bw, bh, width, height)
                    area = bbox[2] * bbox[3]
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": cls_idx + 1,
                            "bbox": [round(v, 2) for v in bbox],
                            "area": round(area, 2),
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )
                    ann_id += 1
            except Exception:
                pass

        image_id += 1

    coco = {
        "info": {"description": f"COCO annotations converted from YOLO for split '{split_dir.name}'"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    out_json.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return out_json


def main() -> None:
    """CLI entry-point to convert all available splits."""
    data_cfg = load_data_yaml(DATA_YAML)
    class_names = data_cfg.get("names", [])
    if not class_names:
        raise SystemExit("No class names found in data/data.yaml -> 'names'.")

    for split in SPLITS:
        split_dir = ROOT / "data" / "raw" / split
        if not split_dir.exists():
            continue
        convert_split(split_dir, class_names)


if __name__ == "__main__":
    main()


