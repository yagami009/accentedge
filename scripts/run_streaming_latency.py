#!/usr/bin/env python3
"""
P0-B — Baseline B: Streaming Latency Pipeline

Runs the streaming pipeline for a configurable duration, capturing from
microphone, converting through Seed-VC, and playing back in real-time.
Saves latency metrics to results/streaming/run_<N>/.

Usage:
    python scripts/run_streaming_latency.py
    python scripts/run_streaming_latency.py --duration 30
    python scripts/run_streaming_latency.py --chunk-ms 100
    python scripts/run_streaming_latency.py --reference samples/reference/us_english_reference_01.wav
"""

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _ram_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def _next_run_id(results_dir: Path) -> str:
    existing = list(results_dir.glob("run_*"))
    if not existing:
        return "run_001"
    nums = [int(d.name.split("_")[1]) for d in existing]
    return f"run_{max(nums) + 1:03d}"


def _parse_args():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="P0-B: Streaming latency pipeline — mic → Seed-VC → speaker"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="How long to run in seconds (default: 10)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=200,
        help="Chunk duration in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate in Hz (default: 22050, must match converter)",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=int,
        default=10,
        help="Crossfade duration in ms (default: 10)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="samples/reference/us_english_reference_01.wav",
        help="Path to reference audio for target voice style",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=10,
        help="Number of diffusion steps (default: 10, lower=faster)",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        default=False,
        help="Enable Voice Activity Detection mode (default: False)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD speech probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=float,
        default=250,
        help="Minimum speech segment duration in ms (default: 250)",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=float,
        default=100,
        help="Minimum silence gap between segments in ms (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: cuda, mps, cpu (default: auto-detect)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    print("=" * 60)
    print("P0-B — Streaming Latency Pipeline")
    print("=" * 60)
    print()

    results_base = PROJECT_ROOT / "results" / "streaming"
    results_base.mkdir(parents=True, exist_ok=True)
    run_id = _next_run_id(results_base)
    run_dir = results_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_log = []

    def log(msg):
        line = f"[{_timestamp()}] {msg}"
        print(line)
        run_log.append(line)

    log(f"Run ID: {run_id}")
    log(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}")
    log(f"Duration: {args.duration}s, Chunk: {args.chunk_ms}ms, Sample rate: {args.sample_rate}Hz")
    log(f"Diffusion steps: {args.diffusion_steps}, Device: {args.device or 'auto'}")

    # ── Load reference audio ─────────────────────────────────────
    reference_path = Path(args.reference)
    if not reference_path.is_absolute():
        reference_path = PROJECT_ROOT / reference_path

    log(f"Loading reference: {reference_path}")
    if not reference_path.exists():
        log(f"ERROR: Reference audio not found: {reference_path}")
        return 1

    ref_data, ref_sr = sf.read(str(reference_path), dtype="float32")
    if len(ref_data.shape) > 1:
        ref_data = ref_data.mean(axis=1)
    ref_duration_s = len(ref_data) / ref_sr
    log(f"  Duration: {ref_duration_s:.2f}s, SR: {ref_sr}Hz")

    # ── Initialize converter ─────────────────────────────────────
    log("Initializing SeedVCConverter...")
    t0 = time.perf_counter()
    from src.conversion.seedvc import SeedVCConverter
    converter = SeedVCConverter(device=args.device or "cpu", model_config="seed-uvit-tat-xlsr-tiny")
    init_time = time.perf_counter() - t0
    log(f"  Init time: {init_time:.2f}s")

    # ── Load model ──────────────────────────────────────────���────
    log("Loading model...")
    t_load_start = time.perf_counter()
    converter._load_model()
    t_load_end = time.perf_counter()
    model_load_time = t_load_end - t_load_start
    log(f"  Model load: {model_load_time:.2f}s")

    ram_before = _ram_mb()
    log(f"  RAM before streaming: {ram_before:.1f} MB")

    # ── Initialize streaming pipeline ────────────────────────────
    log("Initializing streaming pipeline...")
    from src.streaming.pipeline import PipelineConfig, StreamingPipeline

    config = PipelineConfig(
        chunk_duration_ms=args.chunk_ms,
        sample_rate=args.sample_rate,
        crossfade_ms=args.crossfade_ms,
        use_vad=args.vad,
        vad_speech_threshold=args.vad_threshold,
        vad_min_speech_ms=args.min_speech_ms,
        vad_min_silence_ms=args.min_silence_ms,
    )
    pipeline = StreamingPipeline(converter=converter, config=config)
    log(f"  Chunk samples: {pipeline.chunk_samples}")
    log(f"  Crossfade samples: {pipeline.crossfade_samples}")

    # ── Run streaming pipeline ───────────────────────────────────
    log(f"Running streaming pipeline for {args.duration}s...")
    log("  Speak into your microphone...")

    try:
        report = pipeline.run(
            reference_audio=ref_data,
            duration_s=args.duration,
        )
        success = True
    except KeyboardInterrupt:
        log("  Interrupted by user")
        success = False
    except Exception as e:
        log(f"  ERROR: {e}")
        import traceback
        log(f"  {traceback.format_exc()}")
        success = False
    finally:
        pipeline.stop()

    # ── Collect metrics ──────────────────────────────────────────
    ram_after = _ram_mb()
    latency_report = pipeline.latency_measurer.report()
    summary = pipeline.latency_measurer.summary()
    log(f"Latency report:\n{summary}")

    # Collect VAD stats if applicable
    vad_stats = pipeline.vad_stats if args.vad else {}
    if vad_stats:
        total_speech = vad_stats.get("total_speech_ms", 0)
        segments = vad_stats.get("segments_found", 0)
        log(f"[VAD] Stats: {segments} segments, {total_speech:.0f}ms total speech")

    # ── Build metrics JSON ───────────────────────────────────────
    metrics = {
        "run_id": run_id,
        "timestamp": _timestamp(),
        "mode": "streaming",
        "model": "seed-uvit-tat-xlsr-tiny",
        "device": args.device or "cpu",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "duration_seconds": args.duration,
        "chunk_duration_ms": args.chunk_ms,
        "sample_rate": args.sample_rate,
        "crossfade_ms": args.crossfade_ms,
        "vad_enabled": args.vad,
        "vad_threshold": args.vad_threshold,
        "vad_min_speech_ms": args.min_speech_ms,
        "vad_min_silence_ms": args.min_silence_ms,
        "diffusion_steps": args.diffusion_steps,
        "reference_audio": str(reference_path),
        "reference_duration_seconds": round(ref_duration_s, 3),
        "model_load_seconds": round(model_load_time, 3),
        "init_seconds": round(init_time, 3),
        "ram_before_mb": round(ram_before, 1),
        "ram_after_mb": round(ram_after, 1),
        "success": success,
    }

    if report and "total_chunks" in report:
        metrics["chunks_processed"] = report["total_chunks"]
        metrics["latency"] = latency_report
        for stage in ["capture", "conversion", "playback", "total"]:
            s = latency_report.get(stage, {})
            metrics[f"{stage}_mean_ms"] = round(s.get("mean", 0), 2)
            metrics[f"{stage}_p50_ms"] = round(s.get("p50", 0), 2)
            metrics[f"{stage}_p95_ms"] = round(s.get("p95", 0), 2)
            metrics[f"{stage}_max_ms"] = round(s.get("max", 0), 2)
    else:
        metrics["chunks_processed"] = 0
        metrics["latency"] = {"error": "No data collected"}

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Metrics saved: {metrics_path}")

    # ── Save run log ─────────────────────────────────────────────
    log_path = run_dir / "run.log"
    with open(log_path, "w") as f:
        f.write("\n".join(run_log))
    log(f"Run log saved: {log_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STREAMING RUN SUMMARY")
    print("=" * 60)
    if success and "total_chunks" in metrics.get("latency", {}):
        print(f"  Status:            SUCCESS")
        print(f"  Chunks processed:  {metrics.get('chunks_processed', 0)}")
        print(f"  Model load time:   {model_load_time:.2f}s")
        if "total" in latency_report:
            t = latency_report["total"]
            print(f"  Avg total latency: {t.get('mean', 0):.1f}ms (P95: {t.get('p95', 0):.1f}ms)")
        if "conversion" in latency_report:
            c = latency_report["conversion"]
            print(f"  Avg conversion:    {c.get('mean', 0):.1f}ms (P95: {c.get('p95', 0):.1f}ms)")
        if "capture" in latency_report:
            cp = latency_report["capture"]
            print(f"  Avg capture:       {cp.get('mean', 0):.1f}ms (P95: {cp.get('p95', 0):.1f}ms)")
        if "playback" in latency_report:
            p = latency_report["playback"]
            print(f"  Avg playback:      {p.get('mean', 0):.1f}ms (P95: {p.get('p95', 0):.1f}ms)")
        print(f"  Peak RAM:          {ram_after:.1f} MB")
        print(f"  Results dir:       {run_dir}")
        if args.vad and vad_stats:
            print(f"  VAD segments:      {vad_stats.get('segments_found', 0)}")
            print(f"  VAD speech:        {vad_stats.get('total_speech_ms', 0):.0f}ms")
    else:
        print(f"  Status:            FAILED or incomplete")
        print(f"  See log:           {log_path}")
    print("=" * 60)

    # ── Cleanup ──────────────────────────────────────────────────
    converter.unload()
    gc.collect()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
