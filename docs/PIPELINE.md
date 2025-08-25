## Pipeline

### Stages
- convert: `python src/convert_pdfs.py` → `data/raw/images`, `reports/convert_log.txt`
- train: `python src/train.py` → `models/best.pt`, `models/best.onnx`, metrics in `reports/`

### Parameters
- `params.yaml` contains `conversion`, `training`, and `rfdetr` sections.

### Data tracking
- Use DVC to track `data/pdfs` and `data/raw/{train,valid,test}`.
- Avoid overlapping outs with stage outputs (keep `data/raw/images` as stage out only).

### Experiments and queuing
- Queue: `python -m dvc exp run --queue -S training.epochs=50`
- Start: `python -m dvc queue start`
- Inspect: `python -m dvc exp show`
- Apply: `python -m dvc exp apply <exp_id>`


