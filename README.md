## my-yolo-project

End-to-end object detection pipeline on PDF-derived images with Ultralytics YOLO and DVC.

### Features
- PDF to image conversion with pypdfium2 (BSD-3) and Pillow
- Manual YOLO-format annotation and dataset split utilities
- Training with Ultralytics YOLOv8/YOLO11 or Roboflow RFDETR, DVCLive metrics, and experiment tracking via DVC
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

### Parameters
Edit `params.yaml`:
- conversion: `dpi`, `format`, `color_mode`, `resize`
- training: `model`, `epochs`, `batch`, `imgsz`, `optimizer`, `patience`, `cache`, `multi_scale`, `workers`, `device`

### Convert PDFs
Place PDFs in `data/pdfs/`, then:
```bash
python src/convert_pdfs.py
```
Outputs images to `data/raw/images` and logs to `reports/convert_log.txt`.

Best practices:
- 300 DPI preserves fine details; PNG avoids artifacts
- Grayscale (`L`) if color is irrelevant
- Convert per page for scalability; validate outputs

Licensing note: pypdfium2 is BSD-3 licensed and suitable for commercial use.

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


