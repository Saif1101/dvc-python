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

    model = YOLO(model_name)

    # Setup DVCLive to collect metrics
    with Live(dir=str(REPORTS_DIR), resume=True) as live:
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

        # Copy best model to models/best.pt
        try:
            best_pt = Path(results.save_dir) / "weights" / "best.pt"
            if best_pt.exists():
                target = MODELS_DIR / "best.pt"
                target.write_bytes(best_pt.read_bytes())
        except Exception:
            pass


if __name__ == "__main__":
    main()


