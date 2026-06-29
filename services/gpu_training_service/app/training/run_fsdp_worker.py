# app/training/run_fsdp_worker.py
from __future__ import annotations

import os
import json
from dataclasses import asdict
from pathlib import Path

import torch

from app.training.fsdp_trainer import FSdpTrainConfig, run_fsdp_training
from app.config.gpu_logger import gpu_info, gpu_warning


def main():
    # torchrun sets these
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    cfg_json = os.environ.get("FSDP_TRAIN_CONFIG_JSON", "")
    if not cfg_json:
        raise RuntimeError("Missing env var FSDP_TRAIN_CONFIG_JSON")

    cfg_dict = json.loads(cfg_json)
    cfg = FSdpTrainConfig(**cfg_dict)

    # Make sure output dir exists
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    gpu_info(
        f"[FSDP-WORKER] rank={rank} local_rank={local_rank} world_size={world_size} "
        f"cuda_available={torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    run_fsdp_training(cfg)


if __name__ == "__main__":
    main()
