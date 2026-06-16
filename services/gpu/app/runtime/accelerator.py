# app/runtime/accelerator.py

import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger("gpu_service")


class AcceleratorRuntime:
    """
    Service-owned accelerator runtime.

    Responsibilities:
    - Discover ALL GPUs once at startup
    - Classify GPUs by memory size
    - Assign GPUs to workload roles (static, deterministic)
    - Expose stable device routing
    - Never make runtime memory guesses
    """

    # ----------------------------
    # Public workload roles
    # ----------------------------
    ROLE_GENERATION = "generation"
    ROLE_EMBEDDING = "embedding"
    ROLE_FALLBACK = "fallback"

    def __init__(self):
        self._gpus = self._discover_gpus()
        self._role_map: Dict[str, torch.device] = {}
        self._primary_device: torch.device = torch.device("cpu")

        if not self._gpus:
            logger.warning("[GPU] CUDA not available — running in CPU mode")
            return

        self._assign_roles()

        logger.info(
            "[GPU] Accelerator initialized | GPUs=%d | roles=%s",
            len(self._gpus),
            {k: str(v) for k, v in self._role_map.items()},
        )

    # ---------------------------------------------------------
    # GPU Discovery
    # ---------------------------------------------------------
    def _discover_gpus(self):
        if not torch.cuda.is_available():
            return []

        gpus = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append({
                "index": idx,
                "name": props.name,
                "total_mem": props.total_memory,  # bytes
            })

        # Sort DESC by memory size
        gpus.sort(key=lambda g: g["total_mem"], reverse=True)

        for g in gpus:
            logger.info(
                "[GPU] Detected GPU %d: %s | %.1f GB",
                g["index"],
                g["name"],
                g["total_mem"] / (1024 ** 3),
            )

        return gpus

    # ---------------------------------------------------------
    # Role Assignment (STATIC)
    # ---------------------------------------------------------
    def _assign_roles(self):
        """
        Deterministic role assignment:

        - Largest GPU → generation
        - Smallest suitable GPU → embedding
        - All others → unused (future expansion)
        """

        # Largest GPU → generation
        gen_gpu = self._gpus[0]
        self._role_map[self.ROLE_GENERATION] = torch.device(
            f"cuda:{gen_gpu['index']}"
        )
        self._primary_device = self._role_map[self.ROLE_GENERATION]

        # Embedding GPU selection
        if len(self._gpus) > 1:
            emb_gpu = self._gpus[-1]  # smallest GPU
        else:
            emb_gpu = gen_gpu  # single-GPU system

        self._role_map[self.ROLE_EMBEDDING] = torch.device(
            f"cuda:{emb_gpu['index']}"
        )

        logger.info(
            "[GPU] Assigned roles | generation=cuda:%d | embedding=cuda:%d",
            gen_gpu["index"],
            emb_gpu["index"],
        )

    # ---------------------------------------------------------
    # Public API (Service Internal)
    # ---------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """
        Backwards compatibility:
        - Returns PRIMARY device (generation)
        """
        return self._primary_device

    def device_for(self, role: str) -> torch.device:
        """
        Explicit role-based device routing.
        """
        return self._role_map.get(role, self._primary_device)

    @property
    def is_gpu(self) -> bool:
        return bool(self._gpus)

    def status(self) -> dict:
        return {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": len(self._gpus),
            "roles": {
                role: str(device) for role, device in self._role_map.items()
            },
            "primary_device": str(self._primary_device),
        }


# ---------------------------------------------------------
# Singleton instance (service-owned)
# ---------------------------------------------------------
ACCELERATOR = AcceleratorRuntime()
