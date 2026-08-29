"""Hardware profile capture."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    cpu_model: str = "unknown"
    logical_cores: int = 1
    physical_cores: int = 1
    ram_gb: float = 0.0
    gpu_model: str | None = None
    torch_version: str = ""
    backend: str = "cpu"
    os: str = platform.system()

    @classmethod
    def capture(cls) -> "HardwareProfile":
        profile = cls()
        profile.logical_cores = cls._cpu_count_logical()
        profile.physical_cores = cls._cpu_count_physical()
        profile.ram_gb = cls._ram_gb()
        profile.gpu_model = cls._gpu_model()
        profile.torch_version = cls._torch_version()
        profile.backend = cls._backend()
        return profile

    @staticmethod
    def _cpu_count_logical() -> int:
        try:
            import os
            return os.cpu_count() or 1
        except Exception:
            return 1

    @staticmethod
    def _cpu_count_physical() -> int:
        try:
            import os
            return os.cpu_count() or 1
        except Exception:
            return 1

    @staticmethod
    def _ram_gb() -> float:
        try:
            import psutil
            return round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:
            pass
        try:
            import os
            val = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return round(val / 1e9, 1)
        except Exception:
            return 0.0

    @staticmethod
    def _gpu_model() -> str | None:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return None

    @staticmethod
    def _torch_version() -> str:
        try:
            import torch
            return torch.__version__
        except ImportError:
            return ""

    @staticmethod
    def _backend() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

