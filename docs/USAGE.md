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
Also writes `reports/predictions_report.csv` mapping images to predictions.

#### MLflow tracking
- Enable in `params.yaml` under `mlflow.enabled: true`.
- Local tracking (default when `tracking_uri` empty) writes to `./mlruns`.
- To point to a server, set `mlflow.tracking_uri` (e.g., `http://localhost:5000`).
- Start UI locally:
```bash
mlflow ui --backend-store-uri mlruns
```
Key fields used:
- `mlflow.experiment_name`: experiment grouping
- `mlflow.run_name`: defaults to `training.name`
- Artifacts: models and `reports/*.csv|yaml` are logged

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

### End-to-end workflow on Windows
1) Validate PDFs:
```bat
scripts\1_prep_pdfs.bat
```
2) Prepare labeling staging (YOLO default; pass `coco` to choose COCO):
```bat
scripts\2_setup_labeling.bat [yolo|coco]
```
3) Version dataset (interactive or pass args):
```bat
scripts\3_version_dataset.bat [new|extend] [suffix]
```
4) Train and generate reports (direct or queued):
```bat
scripts\4_train_and_report.bat [queue]
```

### Experiments
```bash
python -m dvc exp run --queue -S training.epochs=50
python -m dvc queue start
python -m dvc exp show
```



