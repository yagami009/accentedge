#!/usr/bin/env python3
"""
P0.2 — First Real Offline Conversion

Performs the first real speech-to-speech conversion using Seed-VC.
Measures timing, RTF, memory, and saves all artifacts.
"""

import gc
import json
import os
import sys
import time
import traceback
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


def _system_ram_gb():
    mem = psutil.virtual_memory()
    return mem.used / 1024 / 1024 / 1024, mem.total / 1024 / 1024 / 1024


def _next_run_id(results_dir):
    existing = list(results_dir.glob("run_*"))
    if not existing:
        return "run_001"
    nums = [int(d.name.split("_")[1]) for d in existing]
    return f"run_{max(nums)+1:03d}"


def main():
    print("=" * 60)
    print("P0.2 — First Real Offline Conversion")
    print("=" * 60)
    print()

    source_path = PROJECT_ROOT / "samples" / "source" / "indian_english_01.wav"
    reference_path = PROJECT_ROOT / "samples" / "reference" / "us_english_reference_01.wav"
    results_base = PROJECT_ROOT / "results" / "offline"
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

    # ── Load source audio ────────────────────────────────────────
    log(f"Loading source: {source_path}")
    source_data, source_sr = sf.read(str(source_path), dtype="float32")
    if len(source_data.shape) > 1:
        source_data = source_data.mean(axis=1)
    source_duration_s = len(source_data) / source_sr
    log(f"  Duration: {source_duration_s:.2f}s, SR: {source_sr}Hz, Channels: {source_data.shape[0] if len(source_data.shape) == 1 else source_data.shape[1]}")

    # ── Load reference audio ─────────────────────────────────────
    log(f"Loading reference: {reference_path}")
    ref_data, ref_sr = sf.read(str(reference_path), dtype="float32")
    if len(ref_data.shape) > 1:
        ref_data = ref_data.mean(axis=1)
    ref_duration_s = len(ref_data) / ref_sr
    log(f"  Duration: {ref_duration_s:.2f}s, SR: {ref_sr}Hz")

    # ── Save source/reference copies ─────────────────────────────
    sf.write(str(run_dir / "source.wav"), source_data, source_sr)
    sf.write(str(run_dir / "reference.wav"), ref_data, ref_sr)
    log(f"Saved source.wav and reference.wav to {run_dir}")

    # ── Initialize converter ─────────────────────────────────────
    log("Initializing SeedVCConverter...")
    t0 = time.perf_counter()
    from src.conversion.seedvc import SeedVCConverter
    converter = SeedVCConverter(device="cpu", model_config="seed-uvit-tat-xlsr-tiny")
    init_time = time.perf_counter() - t0
    log(f"  Init time: {init_time:.2f}s")

    # ── Model loading ────────────────────────────────────────────
    ram_before = _ram_mb()
    sys_ram_before, sys_ram_total = _system_ram_gb()

    log("Loading model...")
    t_load_start = time.perf_counter()
    converter._load_model()
    t_load_end = time.perf_counter()
    model_load_time = t_load_end - t_load_start

    ram_after_load = _ram_mb()
    sys_ram_after, _ = _system_ram_gb()

    log(f"  Model load: {model_load_time:.2f}s")
    log(f"  RAM: {ram_before:.1f} MB -> {ram_after_load:.1f} MB (delta: {ram_after_load - ram_before:.1f} MB)")
    log(f"  System RAM: {sys_ram_before:.2f} GB -> {sys_ram_after:.2f} GB")

    # ── Run conversion ───────────────────────────────────────────
    log("Running conversion...")
    t_conv_start = time.perf_counter()
    try:
        result = converter.convert(
            source_audio=source_data,
            reference_audio=ref_data,
            output_path=run_dir / "output.wav",
            diffusion_steps=10,
        )
        t_conv_end = time.perf_counter()
        conversion_time = t_conv_end - t_conv_start

        output_sr = result.get("sample_rate", source_sr)
        output_duration_s = result.get("output_duration_ms", 0) / 1000.0
        rtf = result.get("rtf", conversion_time / source_duration_s if source_duration_s > 0 else 0)

        log(f"  Conversion time: {conversion_time:.2f}s")
        log(f"  Input duration: {source_duration_s:.2f}s")
        log(f"  Output duration: {output_duration_s:.2f}s")
        log(f"  RTF: {rtf:.3f}")
        log(f"  Output sample rate: {output_sr}Hz")
        log(f"  Output saved: {run_dir / 'output.wav'}")

    except Exception as e:
        log(f"  CONVERSION FAILED: {e}")
        log(f"  {traceback.format_exc()}")
        result = None
        conversion_time = time.perf_counter() - t_conv_start
        rtf = None
        output_duration_s = None
        output_sr = None

    # ── Memory after conversion ──────────────────────────────────
    ram_after_conv = _ram_mb()
    sys_ram_after_conv, _ = _system_ram_gb()

    # ── Unload ───────────────────────────────────────────────────
    log("Unloading model...")
    converter.unload()
    gc.collect()
    ram_final = _ram_mb()
    log(f"  Final RAM: {ram_final:.1f} MB")

    # ── Build metrics JSON ───────────────────────────────────────
    metrics = {
        "run_id": run_id,
        "timestamp": _timestamp(),
        "model": "seed-uvit-tat-xlsr-tiny",
        "device": "cpu",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "input_duration_seconds": round(source_duration_s, 3),
        "reference_duration_seconds": round(ref_duration_s, 3),
        "model_load_seconds": round(model_load_time, 3),
        "conversion_seconds": round(conversion_time, 3),
        "rtf": round(rtf, 4) if rtf is not None else None,
        "peak_ram_mb": round(ram_after_load, 1),
        "ram_after_conversion_mb": round(ram_after_conv, 1),
        "ram_final_mb": round(ram_final, 1),
        "peak_system_ram_gb": round(sys_ram_after, 2),
        "input_sample_rate": source_sr,
        "output_sample_rate": output_sr,
        "input_channels": 1,
        "output_channels": 1,
        "output_duration_seconds": round(output_duration_s, 3) if output_duration_s else None,
        "diffusion_steps": 10,
        "success": result is not None and (run_dir / "output.wav").exists(),
    }

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
    print("CONVERSION SUMMARY")
    print("=" * 60)
    if metrics["success"]:
        print(f"  Status:          SUCCESS")
        print(f"  Output:          {run_dir / 'output.wav'}")
        print(f"  Input duration:  {source_duration_s:.2f}s")
        print(f"  Output duration: {output_duration_s:.2f}s")
        print(f"  Model load:      {model_load_time:.2f}s")
        print(f"  Conversion:      {conversion_time:.2f}s")
        print(f"  RTF:             {rtf:.4f}")
        print(f"  Peak RAM:        {ram_after_load:.1f} MB")
    else:
        print(f"  Status:          FAILED")

    print("=" * 60)
    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
