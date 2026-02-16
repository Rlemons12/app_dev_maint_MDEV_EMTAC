import os
import requests
import logging
import torch

log = logging.getLogger("gpu_training_adapter")


class GPUTrainingAdapter:
    """
    GPU service adapter for MODEL_TRAINING.

    Priority order:
        1) Remote GPU service (HTTP)
        2) Local CUDA
        3) CPU fallback
    """

    def __init__(self):
        self.gpu_service_url = os.getenv("GPU_SERVICE_URL")
        self.enabled = bool(self.gpu_service_url)
        self.remote_available = False

        if self.enabled:
            self.remote_available = self._check_gpu_service()

        self.local_cuda = torch.cuda.is_available()

        log.info(
            "[GPU-ADAPTER] Initialized | remote=%s | local_cuda=%s | torch_cuda=%s",
            self.remote_available,
            self.local_cuda,
            torch.version.cuda,
        )

    def is_available(self) -> bool:
        """
        Returns True if REMOTE GPU SERVICE is available.

        This is intentionally strict:
        - True  → delegate training via HTTP
        - False → run locally (CUDA or CPU)
        """
        return self.remote_available
    # --------------------------------------------------
    # Health checks
    # --------------------------------------------------
    def _check_gpu_service(self) -> bool:
        try:
            r = requests.get(f"{self.gpu_service_url}/health", timeout=2)
            if r.status_code == 200:
                log.info("[GPU-ADAPTER] GPU service online: %s", r.json())
                return True
        except Exception as exc:
            log.warning("[GPU-ADAPTER] GPU service unavailable: %s", exc)
        return False

    # --------------------------------------------------
    # Device resolution
    # --------------------------------------------------
    def get_device(self, *, prefer_local: bool = False) -> torch.device:
        """
        prefer_local=True → use local CUDA even if remote GPU service exists
        """
        if prefer_local and self.local_cuda:
            return torch.device("cuda")

        if self.remote_available:
            return torch.device("cpu")

        if self.local_cuda:
            return torch.device("cuda")

        return torch.device("cpu")

    # --------------------------------------------------
    # Model preparation
    # --------------------------------------------------
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Moves model to appropriate device OR registers
        it with remote GPU service.
        """
        if self.remote_available:
            log.info("[GPU-ADAPTER] Mode=REMOTE (CPU tensors, GPU via service)")
            self._register_remote_model(model)
            return model  # stays CPU-side

        device = self.get_device()
        log.info("[GPU-ADAPTER] Mode=LOCAL device=%s", device)
        return model.to(device)

    def _register_remote_model(self, model):
        """
        Optional hook:
        - future: serialize weights
        - future: stream gradients
        """
        # For now, training still happens locally,
        # but this hook allows evolution.
        log.info("[GPU-ADAPTER] Remote model registration placeholder")

    # --------------------------------------------------
    # Trainer wrapping (future-safe)
    # --------------------------------------------------
    def wrap_trainer(self, trainer):
        """
        Placeholder for:
        - Distributed training
        - Gradient offloading
        - RPC-backed backward()
        """
        return trainer

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    def describe(self) -> dict:
        return {
            "gpu_service_url": self.gpu_service_url,
            "remote_available": self.remote_available,
            "local_cuda": self.local_cuda,
            "torch_cuda_version": torch.version.cuda,
        }
