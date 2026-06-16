# configuration/mlflow_utils.py
from __future__ import annotations

import os
import json
import socket
import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("mlflow_utils")


def _safe_json(d: Dict[str, Any]) -> str:
    try:
        return json.dumps(d, default=str)
    except Exception:
        return str(d)


def init_mlflow(
    experiment_name: str,
    run_name: str,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> "mlflow.ActiveRun":
    """
    Initializes MLflow and starts a run.

    tracking_uri:
      - If None: uses env MLFLOW_TRACKING_URI if set.
      - If still None: MLflow defaults (file-based ./mlruns).

    Returns the active run object.
    """
    import mlflow  # local import so script still runs if MLflow isn't installed

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(experiment_name)

    # Standard tags
    base_tags = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    if tags:
        base_tags.update(tags)

    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags({k: str(v) for k, v in base_tags.items()})

    log.info("[MLFLOW] tracking_uri=%s experiment=%s run_name=%s", uri, experiment_name, run_name)
    return run


def log_params(params: Dict[str, Any]) -> None:
    import mlflow
    # MLflow params are string-ish; also capped in some backends
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            mlflow.log_param(k, str(v))


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    import mlflow
    # Only log numeric-ish metrics
    filtered = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            filtered[k] = float(v)
    if not filtered:
        return
    try:
        mlflow.log_metrics(filtered, step=step)
    except TypeError:
        # Older mlflow versions may not accept step with log_metrics
        for k, v in filtered.items():
            mlflow.log_metric(k, v, step=step if step is not None else 0)


def log_artifacts(run_dir: Path, subdir: Optional[str] = None) -> None:
    import mlflow
    p = Path(run_dir)
    if not p.exists():
        return

    # Log the whole run_dir (or a subdir) as artifacts
    target = p if subdir is None else (p / subdir)
    if target.exists():
        mlflow.log_artifacts(str(target))


def log_text(name: str, content: str) -> None:
    import mlflow
    # Write a temp file in cwd and log it
    tmp = Path(f".mlflow_{name}.txt")
    tmp.write_text(content, encoding="utf-8")
    mlflow.log_artifact(str(tmp))
    try:
        tmp.unlink()
    except Exception:
        pass


def end_mlflow(success: bool = True) -> None:
    import mlflow
    mlflow.end_run(status="FINISHED" if success else "FAILED")
