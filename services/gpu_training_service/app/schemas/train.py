# app/schemas/train.py
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class TrainStartRequest(BaseModel):
    job_name: str = Field(..., description="Friendly name for the training job")
    base_model_path: str = Field(..., description="Local path to HF model (offline-safe)")
    output_dir: str = Field(..., description="Where checkpoints/logs will be written")

    # Data: simplest possible: a local JSONL of {"text": "..."} or {"prompt": "...", "response": "..."}
    train_data_path: str = Field(..., description="Local path to training data file")
    eval_data_path: Optional[str] = Field(None, description="Optional local path to eval data file")

    # Training hyperparams
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    # System
    mixed_precision: str = Field("bf16", description="bf16|fp16|no")
    seed: int = 42

    # FSDP configuration
    fsdp_sharding_strategy: str = Field(
        "FULL_SHARD",
        description="FULL_SHARD|SHARD_GRAD_OP|NO_SHARD (maps to torch.distributed.fsdp.ShardingStrategy)"
    )
    fsdp_auto_wrap_policy: str = Field(
        "transformer",
        description="transformer|size_based"
    )
    fsdp_min_num_params: int = Field(100_000_000, description="Used for size_based auto wrap policy")

    # Launch
    nproc_per_node: Optional[int] = Field(
        None,
        description="How many GPU processes to launch. Default: all visible GPUs"
    )

    extra_env: Dict[str, str] = Field(default_factory=dict)
    extra_args: List[str] = Field(default_factory=list, description="Extra args forwarded to worker script")


class TrainStartResponse(BaseModel):
    job_id: str
    status: str
    message: str
    launch_cmd: List[str]


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    return_code: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    output_dir: Optional[str] = None
    log_file: Optional[str] = None
    last_lines: List[str] = []
    error: Optional[str] = None
