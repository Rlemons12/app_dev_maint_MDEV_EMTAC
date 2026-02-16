# app/training/fsdp_trainer.py
from __future__ import annotations

import os
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterable

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config.gpu_logger import gpu_info, gpu_warning, gpu_error, gpu_snapshot


# ----------------------------
# Config
# ----------------------------
@dataclass
class FSdpTrainConfig:
    job_name: str
    base_model_path: str
    output_dir: str
    train_data_path: str
    eval_data_path: Optional[str] = None

    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    mixed_precision: str = "bf16"  # bf16|fp16|no
    seed: int = 42

    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_auto_wrap_policy: str = "transformer"
    fsdp_min_num_params: int = 100_000_000


# ----------------------------
# Data
# ----------------------------
class JsonlTextDataset(Dataset):
    """
    Expects JSONL lines with:
      {"text": "..."} OR {"prompt": "...", "response": "..."}
    Produces a single concatenated text field.
    """

    def __init__(self, path: str):
        self.rows: List[Dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"Dataset not found: {path}")

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

        if not self.rows:
            raise RuntimeError(f"Empty dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> str:
        r = self.rows[idx]
        if "text" in r:
            return str(r["text"])
        prompt = str(r.get("prompt", ""))
        response = str(r.get("response", ""))
        # Minimal supervised format
        return (prompt + "\n" + response).strip()


def _seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Distributed init
# ----------------------------
def _init_dist():
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _is_main() -> bool:
    return _rank() == 0


# ----------------------------
# FSDP helpers
# ----------------------------
def _parse_sharding_strategy(s: str) -> ShardingStrategy:
    s = (s or "").upper().strip()
    if s == "FULL_SHARD":
        return ShardingStrategy.FULL_SHARD
    if s == "SHARD_GRAD_OP":
        return ShardingStrategy.SHARD_GRAD_OP
    if s == "NO_SHARD":
        return ShardingStrategy.NO_SHARD
    raise ValueError(f"Unknown sharding strategy: {s}")


def _build_mixed_precision(mode: str) -> Optional[MixedPrecision]:
    mode = (mode or "").lower().strip()
    if mode == "no":
        return None
    if mode == "bf16":
        dtype = torch.bfloat16
    elif mode == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("mixed_precision must be bf16|fp16|no")

    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def _build_auto_wrap_policy(model, cfg: FSdpTrainConfig):
    ap = (cfg.fsdp_auto_wrap_policy or "").lower().strip()
    if ap == "size_based":
        return size_based_auto_wrap_policy(min_num_params=int(cfg.fsdp_min_num_params))

    # "transformer" default:
    # Try to wrap transformer layers. This works broadly for HF causal LMs.
    try:
        import transformers
        layer_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
        return transformer_auto_wrap_policy(transformer_layer_cls={layer_cls})
    except Exception:
        # fallback to size-based if we can't import a known layer class
        return size_based_auto_wrap_policy(min_num_params=int(cfg.fsdp_min_num_params))


# ----------------------------
# Training loop
# ----------------------------
def run_fsdp_training(cfg: FSdpTrainConfig):
    _seed_everything(cfg.seed)
    _init_dist()

    rank = _rank()
    world = _world()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    if _is_main():
        gpu_info(
            f"[FSDP] Starting | job={cfg.job_name} world={world} "
            f"model={cfg.base_model_path} out={cfg.output_dir}"
        )

    # ----------------------------
    # Load tokenizer/model (CPU first to reduce GPU spikes)
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_snapshot("[FSDP] before-model-load")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if cfg.mixed_precision == "bf16" else None,
        trust_remote_code=True,
    )

    model.train()

    # ----------------------------
    # Wrap with FSDP
    # ----------------------------
    sharding = _parse_sharding_strategy(cfg.fsdp_sharding_strategy)
    mp = _build_mixed_precision(cfg.mixed_precision)

    auto_wrap = _build_auto_wrap_policy(model, cfg)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=sharding,
        mixed_precision=mp,
        device_id=device if device.type == "cuda" else None,
    )

    gpu_snapshot("[FSDP] after-fsdp-wrap")

    # ----------------------------
    # DataLoader
    # ----------------------------
    train_ds = JsonlTextDataset(cfg.train_data_path)

    def collate(batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=int(cfg.max_seq_length),
            return_tensors="pt",
        )
        # Causal LM labels = input_ids with ignore_index for pad
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return {k: v.to(device) for k, v in enc.items()}

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.per_device_train_batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    # ----------------------------
    # Optimizer + schedule
    # ----------------------------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    total_steps = int(math.ceil(len(train_loader) * float(cfg.num_train_epochs)))
    warmup_steps = int(total_steps * float(cfg.warmup_ratio))

    def lr_for_step(step: int) -> float:
        if warmup_steps <= 0:
            return float(cfg.learning_rate)
        if step < warmup_steps:
            return float(cfg.learning_rate) * (step / max(1, warmup_steps))
        return float(cfg.learning_rate)

    # ----------------------------
    # Train
    # ----------------------------
    global_step = 0
    grad_accum = max(1, int(cfg.gradient_accumulation_steps))

    for epoch in range(int(math.ceil(float(cfg.num_train_epochs)))):
        if _is_main():
            gpu_info(f"[FSDP] Epoch {epoch+1} starting")

        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                global_step += 1

                # LR schedule (simple warmup)
                lr = lr_for_step(global_step)
                for pg in opt.param_groups:
                    pg["lr"] = lr

                opt.step()
                opt.zero_grad(set_to_none=True)

                if global_step % 10 == 0:
                    # Only main logs to avoid spam
                    if _is_main():
                        gpu_info(f"[FSDP] step={global_step}/{total_steps} loss={loss.item():.4f} lr={lr:.2e}")

        # ----------------------------
        # Save checkpoint (main rank only)
        # ----------------------------
        if _is_main():
            ckpt_dir = Path(cfg.output_dir) / f"checkpoint-epoch-{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Save full state dict (simple + reliable)
            # For very large models you may want sharded state dict saves.
            gpu_info(f"[FSDP] Saving checkpoint to {ckpt_dir}")

            # unwrap is not required; FSDP provides state_dict
            sd = model.state_dict()
            torch.save(sd, ckpt_dir / "pytorch_model_fsdp.pt")
            tokenizer.save_pretrained(str(ckpt_dir))

    if _is_main():
        gpu_info("[FSDP] Training complete")

    dist.barrier()
    dist.destroy_process_group()
