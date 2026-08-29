"""Smoke test for Streaming AC candidate."""

from __future__ import annotations

import inspect

from accentedge_lab.models.streaming_ac import StreamingACCandidate, StreamingACConfig
from accentedge_lab.models.streaming_ac.low_lookahead import LowLookaheadContentProsodyEncoder
from accentedge_lab.models.streaming_ac.paper_style import (
    ContentProsodyEncoder as PaperStyleContentProsodyEncoder,
)

config = StreamingACConfig(mode="low_lookahead")
print("config.mode:", repr(config.mode))

candidate = StreamingACCandidate(config=config)
print("encoder type:", type(candidate.encoder))
print("encoder class name:", candidate.encoder.__class__.__name__)
print("encoder module:", candidate.encoder.__class__.__module__)

cls = candidate.encoder.__class__
print("LowLookaheadContentProsodyEncoder:", LowLookaheadContentProsodyEncoder)
print("cls is LowLookaheadContentProsodyEncoder:", cls is LowLookaheadContentProsodyEncoder)
print("isinstance:", isinstance(candidate.encoder, LowLookaheadContentProsodyEncoder))
print("isinstance paper:", isinstance(candidate.encoder, PaperStyleContentProsodyEncoder))

# Also verify _build source text
src = inspect.getsource(StreamingACCandidate._build)
print("_build source lines:")
for i, line in enumerate(src.splitlines()[:25], 1):
    print(f"{i}: {line}")
