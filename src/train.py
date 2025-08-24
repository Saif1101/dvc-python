from __future__ import annotations

from pathlib import Path
import json
import yaml
import pandas as pd
from ultralytics import YOLO
from dvclive import Live


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
DATA_YAML = ROOT / "data" / "data.yaml"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def load_params(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    params = load_params(PARAMS_PATH)
    train_p = params.get("training", {})
    rfdetr_p = params.get("rfdetr", {})

    model_type = train_p.get("model_type", "ultralytics")
    model_name = train_p.get("model", "yolov8n.pt")
    epochs = int(train_p.get("epochs", 100))
    batch = int(train_p.get("batch", 16))
    imgsz = int(train_p.get("imgsz", 640))
    optimizer = train_p.get("optimizer", "auto")
    patience = int(train_p.get("patience", 50))
    cache = bool(train_p.get("cache", True))
    multi_scale = bool(train_p.get("multi_scale", False))
    workers = int(train_p.get("workers", 4))
    device = train_p.get("device", "") or None
    project = train_p.get("project", str(REPORTS_DIR / "ultralytics"))
    name = train_p.get("name", "exp")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup DVCLive to collect metrics
    with Live(dir=str(REPORTS_DIR), resume=True) as live:
        if model_type == "ultralytics":
            model = YOLO(model_name)
            results = model.train(
                data=str(DATA_YAML),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                optimizer=optimizer,
                patience=patience,
                cache=cache,
                multi_scale=multi_scale,
                workers=workers,
                device=device,
                amp=True,
                project=project,
                name=name,
                exist_ok=True,
            )
        elif model_type == "rfdetr":
            # Lazy import so ultralytics remains optional
            from rfdetr import RFDETRBase

            dataset_dir = rfdetr_p.get("dataset_dir", str(ROOT / "data" / "raw"))
            output_dir = rfdetr_p.get("output_dir", str(REPORTS_DIR / "rfdetr_runs"))
            rf_epochs = int(rfdetr_p.get("epochs", 50))
            rf_bs = int(rfdetr_p.get("batch_size", 4))
            rf_gas = int(rfdetr_p.get("grad_accum_steps", 4))
            rf_lr = float(rfdetr_p.get("lr", 1e-4))

            (Path(output_dir)).mkdir(parents=True, exist_ok=True)
            model = RFDETRBase()
            results = model.train(
                dataset_dir=dataset_dir,
                epochs=rf_epochs,
                batch_size=rf_bs,
                grad_accum_steps=rf_gas,
                lr=rf_lr,
                output_dir=output_dir,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Ultralytics saves metrics to runs; aggregate key metrics to CSV
        metrics_json = results.metrics if hasattr(results, "metrics") else {}
        # Fallback: read results.json in the run directory
        try:
            run_dir = Path(results.save_dir)
            results_json_path = run_dir / "results.json"
            if results_json_path.exists():
                metrics_json = json.loads(results_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

        metrics_csv = REPORTS_DIR / "train_metrics.csv"
        try:
            if isinstance(metrics_json, list):
                df = pd.DataFrame(metrics_json)
            else:
                df = pd.DataFrame([metrics_json])
            df.to_csv(metrics_csv, index=False)
        except Exception:
            # create minimal CSV if parsing failed
            pd.DataFrame([{"status": "completed"}]).to_csv(metrics_csv, index=False)

        # Log key metrics to DVCLive
        try:
            if isinstance(metrics_json, list):
                for step_idx, row in enumerate(metrics_json, start=1):
                    for k, v in row.items():
                        if isinstance(v, (int, float)):
                            live.log_metric(k, v, step=step_idx)
            elif isinstance(metrics_json, dict):
                for k, v in metrics_json.items():
                    if isinstance(v, (int, float)):
                        live.log_metric(k, v, step=epochs)
            live.next_step()
        except Exception:
            pass

        # Save params for traceability
        params_out = REPORTS_DIR / "train_params.yaml"
        with open(params_out, "w", encoding="utf-8") as f:
            yaml.safe_dump({"training": train_p}, f, sort_keys=False)

        # Save/export models
        try:
            if model_type == "ultralytics":
                best_pt = Path(results.save_dir) / "weights" / "best.pt"
                if best_pt.exists():
                    target = MODELS_DIR / "best.pt"
                    target.write_bytes(best_pt.read_bytes())

                    # Optional ONNX export
                    if bool(train_p.get("onnx_export", True)):
                        onnx_path = MODELS_DIR / "best.onnx"
                        model = YOLO(str(target))
                        model.export(
                            format="onnx",
                            opset=int(train_p.get("onnx_opset", 12)),
                            dynamic=bool(train_p.get("onnx_dynamic", True)),
                        )
                        # Ultralytics writes next to weights; move if created
                        run_dir = Path(results.save_dir)
                        candidate = run_dir / "weights" / "best.onnx"
                        if candidate.exists():
                            onnx_path.write_bytes(candidate.read_bytes())
            elif model_type == "rfdetr":
                # RFDETR saves into output_dir; users can convert to ONNX externally if needed
                pass
        except Exception:
            pass


if __name__ == "__main__":
    main()


