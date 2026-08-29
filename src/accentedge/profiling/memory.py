"""Memory profiling utilities."""

from __future__ import annotations

import sys
from typing import Any


def measure_model_memory(model: Any) -> int:
    try:
        total = 0
        for param in model.parameters():
            total += param.numel() * param.element_size()
        for buf in model.buffers():
            total += buf.numel() * buf.element_size()
        return total
    except Exception:
        return 0


def measure_session_memory(session: Any) -> int:
    try:
        if hasattr(session, "state_size_bytes"):
            return session.state_size_bytes()
    except Exception:
        pass
    return 0


def profile_inference_memory(
    candidate: Any, session: Any, chunk: Any
) -> dict[str, int]:
    try:
        model_mem = measure_model_memory(candidate)
        session_mem = measure_session_memory(session)
        return {
            "model_bytes": model_mem,
            "session_bytes": session_mem,
            "total_bytes": model_mem + session_mem,
        }
    except Exception:
        return {"model_bytes": 0, "session_bytes": 0, "total_bytes": 0}
