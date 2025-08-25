## Usage

### Setup
```bash
pip install -r requirements.txt
python -m dvc init
```

### Convert PDFs to images
```bash
python src/convert_pdfs.py
```

### Annotate and split
- Use LabelImg/Roboflow to generate YOLO `.txt` labels.
- If needed, split:
```bash
python -c "from pathlib import Path; from src.utils import split_dataset; p=Path('data/raw'); split_dataset(p/'images', p/'labels', p)"
```

### Train (Ultralytics)
```bash
python src/train.py
```
Produces `models/best.pt` and `models/best.onnx` (if enabled).

### Train (RFDETR)
1) Convert YOLO to COCO:
```bash
python src/yolo_to_coco.py
```
2) Set `training.model_type: rfdetr` in `params.yaml`.
3) Train:
```bash
python src/train.py
```

### DVC pipeline
```bash
python -m dvc repro
```

### Experiments
```bash
python -m dvc exp run --queue -S training.epochs=50
python -m dvc queue start
python -m dvc exp show
```


