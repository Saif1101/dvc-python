from __future__ import annotations

"""Training entry-point supporting Ultralytics YOLO and Roboflow RFDETR.

Behavior is controlled by ``params.yaml``:
- training.model_type: "ultralytics" or "rfdetr"
- Ultralytics settings under ``training`` (epochs, batch, imgsz, etc.)
- RFDETR settings under ``rfdetr`` (dataset_dir, output_dir, epochs, ...)

When Ultralytics is used and ``training.onnx_export: true``, the script exports
``models/best.onnx`` after training for deployment.
"""

from pathlib import Path
import json
import yaml
import pandas as pd
from ultralytics import YOLO
from dvclive import Live
from typing import Any, Dict, Iterable, Optional

# MLflow is optional; guard imports so training still works without it
try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    mlflow = None  # type: ignore
    _MLFLOW_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
DATA_YAML = ROOT / "data" / "data.yaml"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


class MLflowTracker:
    def __init__(self, enabled: bool, config: dict, run_name: Optional[str]) -> None:
        self.enabled = bool(enabled) and _MLFLOW_AVAILABLE
        self.config = config
        self.run_name = run_name
        self._active = False

    def start(self) -> None:
        if not self.enabled:
            return
        tracking_uri = self.config.get("tracking_uri") or None
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        registry_uri = self.config.get("registry_uri") or None
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        experiment_name = self.config.get("experiment_name", "YOLO-DVC")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=self.run_name)
        self._active = True

    def log_params(self, params: dict) -> None:
        if self._active:
            try:
                mlflow.log_params(params)
            except Exception:
                pass

    def set_tags(self, tags: dict) -> None:
        if self._active:
            try:
                mlflow.set_tags(tags)
            except Exception:
                pass

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self._active:
            try:
                mlflow.log_metric(key, float(value), step=step)
            except Exception:
                pass

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        if self._active:
            try:
                if artifact_path:
                    mlflow.log_artifact(path, artifact_path=artifact_path)
                else:
                    mlflow.log_artifact(path)
            except Exception:
                pass

    def log_artifacts(self, dir_path: str, artifact_path: Optional[str] = None) -> None:
        if self._active:
            try:
                if artifact_path:
                    mlflow.log_artifacts(dir_path, artifact_path=artifact_path)
                else:
                    mlflow.log_artifacts(dir_path)
            except Exception:
                pass

    def end(self) -> None:
        if self._active:
            try:
                mlflow.end_run()
            except Exception:
                pass
            self._active = False


class PredictionsReporter:
    def __init__(self, model: YOLO, reports_dir: Path) -> None:
        self.model = model
        self.reports_dir = reports_dir

    def generate(self, train_dir: Path, test_dir: Path, out_csv: Path) -> None:
        import csv

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "image_path", "class_id", "confidence", "x1", "y1", "x2", "y2"])
            self._predict_split("train", train_dir, writer)
            self._predict_split("test", test_dir, writer)

    def _predict_split(self, split_name: str, source: Path, writer: Any) -> None:
        if not source or not source.exists():
            return
        results = self.model.predict(source=str(source), save=False, stream=False, verbose=False)
        for r in results:
            img_path = getattr(r, "path", "")
            try:
                boxes = r.boxes
                if boxes is None:
                    continue
                for b in boxes:
                    xyxy = b.xyxy[0].tolist() if hasattr(b, "xyxy") else [0, 0, 0, 0]
                    cls_id = int(b.cls.item()) if hasattr(b, "cls") else -1
                    conf = float(b.conf.item()) if hasattr(b, "conf") else 0.0
                    writer.writerow([split_name, img_path, cls_id, conf, *xyxy])
            except Exception:
                continue


def load_params(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    params = load_params(PARAMS_PATH)
    train_p = params.get("training", {})
    rfdetr_p = params.get("rfdetr", {})
    mlflow_p = params.get("mlflow", {})

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

    # --- Optional MLflow setup ---
    run_name = (mlflow_p.get("run_name") or name) if "name" in locals() else mlflow_p.get("run_name")
    tracker = MLflowTracker(enabled=mlflow_p.get("enabled", False), config=mlflow_p, run_name=run_name)

    # Setup DVCLive to collect metrics
    with Live(dir=str(REPORTS_DIR), resume=True) as live:
        # Start MLflow run as a sibling tracker
        tracker.start()
        # Log training parameters up front for reproducibility
        try:
            def _prefixed(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
                return {f"{prefix}.{k}": v for k, v in d.items()}

            tracker.log_params({
                **_prefixed(train_p, "training"),
                **_prefixed(rfdetr_p, "rfdetr"),
                "env.device": device or "auto",
                "env.workers": workers,
                "env.ultralytics_version": getattr(YOLO, "__version__", "unknown"),
                "config.model_type": train_p.get("model_type", "ultralytics"),
            })
            tracker.set_tags({
                "framework": "ultralytics" if train_p.get("model_type", "ultralytics") == "ultralytics" else "rfdetr",
                "project": "YOLO-DVC",
            })
        except Exception:
            pass
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
                            tracker.log_metric(k, float(v), step=step_idx)
            elif isinstance(metrics_json, dict):
                for k, v in metrics_json.items():
                    if isinstance(v, (int, float)):
                        live.log_metric(k, v, step=epochs)
                        tracker.log_metric(k, float(v), step=epochs)
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

        # Log artifacts to MLflow at the end
        try:
            if (MODELS_DIR / "best.pt").exists():
                tracker.log_artifact(str(MODELS_DIR / "best.pt"), artifact_path="models")
            if (MODELS_DIR / "best.onnx").exists():
                tracker.log_artifact(str(MODELS_DIR / "best.onnx"), artifact_path="models")
            if (REPORTS_DIR / "train_metrics.csv").exists():
                tracker.log_artifact(str(REPORTS_DIR / "train_metrics.csv"), artifact_path="reports")
            if (REPORTS_DIR / "train_params.yaml").exists():
                tracker.log_artifact(str(REPORTS_DIR / "train_params.yaml"), artifact_path="reports")
            if (REPORTS_DIR / "predictions_report.csv").exists():
                tracker.log_artifact(str(REPORTS_DIR / "predictions_report.csv"), artifact_path="reports")
            # Also capture the native training run directory, if available
            try:
                run_dir = Path(results.save_dir)
                if run_dir.exists():
                    tracker.log_artifacts(str(run_dir), artifact_path="trainer_run_dir")
            except Exception:
                pass
        except Exception:
            pass

        # After training, generate predictions report for train and test splits
        try:
            if model_type == "ultralytics":
                # Load best model if available
                model_path = MODELS_DIR / "best.pt"
                if model_path.exists():
                    pred_model = YOLO(str(model_path))
                else:
                    pred_model = model  # type: ignore[assignment]

                # Read dataset paths
                try:
                    with open(DATA_YAML, "r", encoding="utf-8") as f:
                        data_cfg = yaml.safe_load(f) or {}
                    train_images_dir = Path(data_cfg.get("train", ""))
                    test_images_dir = Path(data_cfg.get("test", ""))
                except Exception:
                    train_images_dir = Path()
                    test_images_dir = Path()

                reporter = PredictionsReporter(pred_model, REPORTS_DIR)
                reporter.generate(train_images_dir, test_images_dir, REPORTS_DIR / "predictions_report.csv")
        except Exception:
            pass

        tracker.end()


if __name__ == "__main__":
    main()


