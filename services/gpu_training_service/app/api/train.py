# app/api/train.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from fastapi import APIRouter, HTTPException

from app.schemas.train import (
    TrainStartRequest,
    TrainStartResponse,
    TrainStatusResponse,
)
from app.training.job_manager import TRAINING_JOBS
from app.config.gpu_logger import gpu_info, gpu_warning, gpu_error, get_request_id


router = APIRouter(prefix="/train", tags=["training"])


def _torchrun_cmd(nproc: int, module_path: str) -> List[str]:
    # Prefer torchrun (PyTorch >=1.10)
    return [
        "torchrun",
        f"--nproc_per_node={nproc}",
        module_path,
    ]


@router.post("/start", response_model=TrainStartResponse)
def start_train(req: TrainStartRequest) -> TrainStartResponse:
    rid = get_request_id()

    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA is not available on this machine.")

    gpu_count = torch.cuda.device_count()
    nproc = int(req.nproc_per_node) if req.nproc_per_node else gpu_count
    if nproc < 1 or nproc > gpu_count:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid nproc_per_node={nproc}. GPU count={gpu_count}.",
        )

    # Validate paths early
    if not Path(req.base_model_path).exists():
        raise HTTPException(status_code=400, detail=f"Model path not found: {req.base_model_path}")
    if not Path(req.train_data_path).exists():
        raise HTTPException(status_code=400, detail=f"Train data not found: {req.train_data_path}")

    out_dir = Path(req.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "train_job.log"

    # Worker module path (python file)
    worker_file = str(Path(__file__).resolve().parents[1] / "training" / "run_fsdp_worker.py")

    # Build config dict passed to workers via env var
    cfg_dict: Dict[str, Any] = req.model_dump()
    cfg_json = json.dumps(cfg_dict)

    env = dict(os.environ)
    env["FSDP_TRAIN_CONFIG_JSON"] = cfg_json

    # Merge extra env
    for k, v in req.extra_env.items():
        env[str(k)] = str(v)

    cmd = _torchrun_cmd(nproc=nproc, module_path=worker_file)
    # Add extra args as passthrough (optional)
    cmd.extend(req.extra_args or [])

    gpu_info(f"[API] /train/start | job_name={req.job_name} nproc={nproc}", rid)

    rec = TRAINING_JOBS.start_job(
        job_name=req.job_name,
        cmd=cmd,
        env=env,
        output_dir=str(out_dir),
        log_file=str(log_file),
    )

    return TrainStartResponse(
        job_id=rec.job_id,
        status=rec.status,
        message="Training job started" if rec.status == "running" else "Training job failed to start",
        launch_cmd=cmd,
    )


@router.get("/status/{job_id}", response_model=TrainStatusResponse)
def train_status(job_id: str, tail_lines: int = 80) -> TrainStatusResponse:
    if job_id not in {j.job_id for j in TRAINING_JOBS.list_jobs()} and job_id not in TRAINING_JOBS._jobs:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    rec = TRAINING_JOBS.status(job_id)
    lines = TRAINING_JOBS.tail_log(rec.log_file, lines=tail_lines)

    def _fmt(ts: Optional[float]) -> Optional[str]:
        if ts is None:
            return None
        import datetime
        return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")

    return TrainStatusResponse(
        job_id=rec.job_id,
        status=rec.status,
        return_code=rec.return_code,
        started_at=_fmt(rec.started_at),
        finished_at=_fmt(rec.finished_at),
        output_dir=rec.output_dir,
        log_file=rec.log_file,
        last_lines=lines,
        error=rec.error,
    )
