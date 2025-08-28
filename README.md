## my-yolo-project

End-to-end object detection pipeline on PDF-derived images with Ultralytics YOLO and DVC.

### Features
- PDF to image conversion with pypdfium2 (BSD-3) and Pillow
- Dataset preparation utilities: duplicate/filename checks with CSV report
- Staging for labeling (YOLO or COCO) with rendered page images
- Versioned datasets (`data/versions/vNNN`) with automated splits and `data.yaml` updates
- Training with Ultralytics YOLOv8/YOLO11 or Roboflow RFDETR
- Metrics via DVCLive and optional MLflow tracking (metrics, params, artifacts)
- Predictions report for train/test splits after training
- Reproducible pipelines with `dvc.yaml`, `params.yaml`, and Git/DVC versioning

### Setup
1. Create and activate a Python 3.10+ environment.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize Git and DVC:
   ```bash
   git init
   dvc init
   ```
4. Add a DVC remote (optional, e.g., S3):
   ```bash
   dvc remote add -d storage s3://your-bucket/your-prefix
   ```

### Data Layout
```
data/
  pdfs/                # input PDFs
  raw/
    images/            # converted images (from PDFs)
    labels/            # optional staging labels before split
    train/valid/test/  # final splits
      images/
      labels/
data/data.yaml         # YOLO dataset config
```

### Documentation
- See `docs/USAGE.md` for common commands
- See `docs/PIPELINE.md` for pipeline stages and experiments
- See `docs/DATA.md` for formats, structure, and DVC tracking
- See `CONTRIBUTING.md` for style and workflow

### Parameters
Edit `params.yaml`:
- conversion: `dpi`, `format`, `color_mode`, `resize`
- training: `model`, `epochs`, `batch`, `imgsz`, `optimizer`, `patience`, `cache`, `multi_scale`, `workers`, `device`
- mlflow: `enabled`, `tracking_uri`, `experiment_name`, `run_name`, `registry_uri`, `log_artifacts`

### End-to-End Workflow (What happens and what you do)
1) Dataset Preparation (you add PDFs; code validates and reports)
   - You: Place PDFs in `data/pdfs/`.
   - Code: Scans PDFs, computes SHA256 to flag duplicates, checks filename validity, writes
     `reports/pdfs_report.csv` and `reports/pdfs_report.txt`.

2) Labeling Staging (you choose convention; code prepares folders and images)
   - You: Choose labeling convention (YOLO or COCO) and label with your tool (e.g., LabelImg).
   - Code: Copies valid PDFs to `data/staging/pdfs/`, renders page images to `data/staging/<format>/images/`,
     and creates empty dirs (`labels/` for YOLO, `annotations/` for COCO).

3) Dataset Versioning (you choose new or extend; code versions and splits)
   - You: Decide to create a new version or extend an existing one (optionally provide a suffix).
   - Code: Creates `data/versions/vNNN` (or `vNNN-suffix`), splits into train/valid/test (70/20/10 by default),
     copies images/labels, and rewrites `data/data.yaml` to point to the new version.

4) Training and Reporting (you run training; code trains, exports, and reports)
   - You: Run `python src/train.py` (or via DVC queue) with `params.yaml` settings.
   - Code: Trains the chosen model, exports `models/best.pt` and optionally `models/best.onnx`,
     writes `reports/train_metrics.csv` and `reports/train_params.yaml`,
     and generates `reports/predictions_report.csv` with predictions for train/test images.
   - If MLflow is enabled: logs params/metrics and all reports/models as artifacts to the MLflow run.

### Quick start on Windows (one-click scripts)
1) Validate PDFs
```bat
scripts\1_prep_pdfs.bat
```
2) Prepare labeling staging (YOLO default; pass `coco` to choose COCO)
```bat
scripts\2_setup_labeling.bat [yolo|coco]
```
3) Version dataset (interactive or pass args)
```bat
scripts\3_version_dataset.bat [new|extend] [suffix]
```
4) Train and generate reports (direct or DVC queue)
```bat
scripts\4_train_and_report.bat [queue]
```

### Annotate
Use LabelImg/Roboflow to create YOLO txt labels for each image in the same basename. Place labels alongside images or import into `data/raw/labels` and split later.

### Split Dataset and Update `data.yaml`
Use utilities in `src/utils.py` to split 70/20/10 into `data/raw/train|valid|test` and update `data/data.yaml` class names.

### Train
Ultralytics (default):
```bash
python src/train.py
```
RFDETR:
1) Ensure COCO-format datasets under `data/raw/{train,valid,test}` with `_annotations.coco.json`.
2) Set in `params.yaml`:
```yaml
training:
  model_type: rfdetr
rfdetr:
  dataset_dir: data/raw
  output_dir: reports/rfdetr_runs
```
3) Run:
```bash
python src/train.py
```
Artifacts:
- `models/best.pt`, `models/best.onnx` (when Ultralytics with ONNX export enabled)
- `reports/train_metrics.csv`, `reports/train_params.yaml`
- Ultralytics run dir under `reports/ultralytics/`
- If MLflow enabled: artifacts also logged to the MLflow run
 - `reports/predictions_report.csv`: predictions for train/test images with columns
   `split,image_path,class_id,confidence,x1,y1,x2,y2`

### DVC Pipeline
```yaml
stages:
  convert: {cmd: python src/convert_pdfs.py, ...}
  train:   {cmd: python src/train.py, ...}
```
Run:
```bash
dvc repro
```

Tracked outputs include models, metrics, training params, and `reports/predictions_report.csv`.

### Experiments
Queue and run:
```bash
dvc exp run --queue -S training.epochs=50 -S conversion.dpi=200
dvc exp run --queue -S training.model=yolov8m.pt
dvc queue start
dvc exp show
```
Apply best:
```bash
dvc exp apply <exp_id>
```

### Versioning
```bash
dvc add data/pdfs data/raw models reports/train_metrics.csv reports/train_params.yaml
git add .
git commit -m "Add data and pipeline"
dvc push
```

### Notes
- For scanned PDFs, consider pre-processing or OCR if needed
- Increase `imgsz` for small objects (document details), watch GPU memory

### MLflow Tracking (optional)
- Enable in `params.yaml` under `mlflow.enabled: true`.
- Local tracking (empty `tracking_uri`) stores runs in `./mlruns`.
- To use a server, set `mlflow.tracking_uri` (e.g., `http://localhost:5000`).
- Start UI locally:
```bash
mlflow ui --backend-store-uri mlruns
```
Logged items: params, metrics, `models/best.pt`, `models/best.onnx` (if export on),
`reports/train_metrics.csv`, `reports/train_params.yaml`, and `reports/predictions_report.csv`.


