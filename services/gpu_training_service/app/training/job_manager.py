# app/training/job_manager.py
from __future__ import annotations

import os
import uuid
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

from app.config.gpu_logger import gpu_info, gpu_warning, gpu_error


@dataclass
class JobRecord:
    job_id: str
    job_name: str
    status: str  # queued|running|finished|failed
    proc: Optional[subprocess.Popen]
    started_at: float
    finished_at: Optional[float]
    return_code: Optional[int]
    output_dir: str
    log_file: str
    error: Optional[str]


class TrainingJobManager:
    def __init__(self):
        self._jobs: Dict[str, JobRecord] = {}

    def _now(self) -> float:
        return time.time()

    def start_job(
        self,
        job_name: str,
        cmd: List[str],
        env: Dict[str, str],
        output_dir: str,
        log_file: str,
    ) -> JobRecord:
        job_id = str(uuid.uuid4())[:8]
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        lf = Path(log_file)
        lf.parent.mkdir(parents=True, exist_ok=True)

        gpu_info(f"[TRAIN] Starting job | job_id={job_id} job_name={job_name}")
        gpu_info(f"[TRAIN] Launch cmd: {' '.join(cmd)}")

        # stdout/stderr -> file
        f = open(lf, "a", encoding="utf-8")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(out_dir),
            )
        except Exception as e:
            gpu_error(f"[TRAIN] Launch failed | job_id={job_id} error={e}")
            f.close()
            rec = JobRecord(
                job_id=job_id,
                job_name=job_name,
                status="failed",
                proc=None,
                started_at=self._now(),
                finished_at=self._now(),
                return_code=None,
                output_dir=str(out_dir),
                log_file=str(lf),
                error=str(e),
            )
            self._jobs[job_id] = rec
            return rec

        rec = JobRecord(
            job_id=job_id,
            job_name=job_name,
            status="running",
            proc=proc,
            started_at=self._now(),
            finished_at=None,
            return_code=None,
            output_dir=str(out_dir),
            log_file=str(lf),
            error=None,
        )
        self._jobs[job_id] = rec
        return rec

    def poll_job(self, job_id: str) -> JobRecord:
        rec = self._jobs[job_id]
        if rec.proc is None:
            return rec

        if rec.status in {"finished", "failed"}:
            return rec

        rc = rec.proc.poll()
        if rc is None:
            return rec

        # finished
        rec.return_code = rc
        rec.finished_at = self._now()
        if rc == 0:
            rec.status = "finished"
            gpu_info(f"[TRAIN] Job finished | job_id={job_id} rc=0")
        else:
            rec.status = "failed"
            rec.error = f"Non-zero return code: {rc}"
            gpu_warning(f"[TRAIN] Job failed | job_id={job_id} rc={rc}")

        self._jobs[job_id] = rec
        return rec

    def status(self, job_id: str, tail_lines: int = 80) -> JobRecord:
        rec = self.poll_job(job_id)
        return rec

    def tail_log(self, log_file: str, lines: int = 80) -> List[str]:
        p = Path(log_file)
        if not p.exists():
            return []
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                data = f.read().splitlines()
            return data[-lines:]
        except Exception:
            return []

    def list_jobs(self) -> List[JobRecord]:
        # refresh running jobs
        for job_id in list(self._jobs.keys()):
            self.poll_job(job_id)
        return list(self._jobs.values())


TRAINING_JOBS = TrainingJobManager()
