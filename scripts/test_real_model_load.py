#!/usr/bin/env python3
"""
P0.1 — Real Model Runtime Validation

Instantiates SeedVCConverter, calls the real lazy model-loading path,
measures memory, confirms components, and unloads.

No mocks. No stubs.
"""

import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _versions():
    versions = {}
    try:
        import torch
        versions["python"] = sys.version
        versions["torch"] = torch.__version__
        versions["cuda_available"] = torch.cuda.is_available()
        versions["mps_available"] = (
            torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
        )
    except Exception as e:
        versions["torch_import_error"] = str(e)
    try:
        import psutil
        versions["psutil"] = psutil.__version__
    except Exception:
        pass
    try:
        import yaml
        versions["pyyaml"] = yaml.__version__
    except Exception:
        pass
    try:
        import soundfile
        versions["soundfile"] = soundfile.__version__
    except Exception:
        pass
    try:
        import librosa
        versions["librosa"] = librosa.__version__
    except Exception:
        pass
    return versions


def _ram_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024


def _system_ram_gb():
    import psutil
    mem = psutil.virtual_memory()
    return mem.used / 1024 / 1024 / 1024, mem.total / 1024 / 1024 / 1024


def _gpu_mb():
    import torch
    info = {"available": False}
    if torch.cuda.is_available():
        info["available"] = True
        info["allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        info["reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
    return info


def _mps_mb():
    import torch
    info = {"available": False}
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        info["available"] = True
        try:
            allocated = torch.mps.get_memory_allocated()
            info["allocated_mb"] = allocated / 1024 / 1024
        except Exception:
            info["allocated_mb"] = 0
    return info


def main():
    print("=" * 60)
    print("AccentEdge Model Load Test")
    print("=" * 60)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "preset": None,
        "device": None,
        "sample_rate": None,
        "seedvc_path": None,
        "config_path": None,
        "checkpoint": None,
        "load_started": None,
        "load_completed": None,
        "load_duration_seconds": None,
        "ram_before_mb": None,
        "ram_after_mb": None,
        "system_ram_before_gb": None,
        "system_ram_after_gb": None,
        "gpu_memory": None,
        "mps_memory": None,
        "components": {},
        "unload": None,
        "success": False,
        "error": None,
    }

    # ── Step 1: Instantiate converter ───────────────────────────────
    print("\n[1] Instantiating SeedVCConverter...")
    try:
        from src.conversion.seedvc import SeedVCConverter, MODEL_PRESETS
    except ImportError as e:
        print(f"  FAIL: Cannot import SeedVCConverter: {e}")
        result["error"] = {"stage": "import", "message": str(e), "traceback": traceback.format_exc()}
        _print_report(result)
        return 1

    try:
        converter = SeedVCConverter(device="cpu")
    except Exception as e:
        print(f"  FAIL: Cannot instantiate SeedVCConverter: {e}")
        result["error"] = {"stage": "init", "message": str(e), "traceback": traceback.format_exc()}
        _print_report(result)
        return 1

    preset_name = converter.model_name
    preset_data = MODEL_PRESETS.get(preset_name, {})

    result["preset"] = preset_name
    result["device"] = converter.device
    result["sample_rate"] = converter.sample_rate
    result["seedvc_path"] = str(converter._seedvc_path) if converter._seedvc_path else None
    result["config_path"] = str(converter._config_path) if converter._config_path else None
    result["checkpoint"] = preset_data.get("checkpoint")

    print(f"  Preset:           {preset_name}")
    print(f"  Seed-VC path:     {converter._seedvc_path}")
    print(f"  Config:           {converter._config_path}")
    print(f"  Checkpoint:       {preset_data.get('checkpoint')}")
    print(f"  Device:           {converter.device}")
    print(f"  Sample rate:      {converter.sample_rate} Hz")

    # ── Step 2: Memory before ───────────────────────────────────────
    print("\n[2] Measuring memory before load...")
    ram_before = _ram_mb()
    sys_ram_before, sys_ram_total = _system_ram_gb()
    gpu_before = _gpu_mb()
    mps_before = _mps_mb()
    print(f"  Process RSS:      {ram_before:.1f} MB")
    print(f"  System RAM:       {sys_ram_before:.2f} / {sys_ram_total:.2f} GB")
    if gpu_before.get("available"):
        print(f"  GPU allocated:    {gpu_before.get('allocated_mb', 0):.1f} MB")
    if mps_before.get("available"):
        print(f"  MPS allocated:    {mps_before.get('allocated_mb', 0):.1f} MB")

    result["ram_before_mb"] = round(ram_before, 1)
    result["system_ram_before_gb"] = round(sys_ram_before, 2)
    result["system_ram_total_gb"] = round(sys_ram_total, 2)
    result["gpu_memory_before"] = gpu_before
    result["mps_memory_before"] = mps_before

    # ── Step 3: Load model ──────────────────────────────────────────
    print("\n[3] Loading model (this may take 20-60s on first run)...")
    t0 = time.perf_counter()
    result["load_started"] = datetime.now(timezone.utc).isoformat()

    try:
        converter._load_model()
    except Exception as e:
        print(f"\n  FAIL: Model loading failed: {e}")
        print(f"\n  Exception type: {type(e).__name__}")
        print(f"\n  Full traceback:")
        traceback.print_exc()

        result["load_completed"] = datetime.now(timezone.utc).isoformat()
        result["load_duration_seconds"] = round(time.perf_counter() - t0, 2)
        result["error"] = {
            "stage": "model_load",
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        _print_report(result)
        return 1

    load_duration = time.perf_counter() - t0
    result["load_completed"] = datetime.now(timezone.utc).isoformat()
    result["load_duration_seconds"] = round(load_duration, 2)

    print(f"  Load duration:    {load_duration:.2f}s")

    # ── Step 4: Memory after ────────────────────────────────────────
    print("\n[4] Measuring memory after load...")
    ram_after = _ram_mb()
    sys_ram_after, _ = _system_ram_gb()
    gpu_after = _gpu_mb()
    mps_after = _mps_mb()
    print(f"  Process RSS:      {ram_after:.1f} MB  (delta: +{ram_after - ram_before:.1f} MB)")
    print(f"  System RAM:       {sys_ram_after:.2f} / {sys_ram_total:.2f} GB")
    if gpu_after.get("available"):
        print(f"  GPU allocated:    {gpu_after.get('allocated_mb', 0):.1f} MB")
    if mps_after.get("available"):
        print(f"  MPS allocated:    {mps_after.get('allocated_mb', 0):.1f} MB")

    result["ram_after_mb"] = round(ram_after, 1)
    result["system_ram_after_gb"] = round(sys_ram_after, 2)
    result["gpu_memory_after"] = gpu_after
    result["mps_memory_after"] = mps_after

    # ── Step 5: Component verification ──────────────────────────────
    print("\n[5] Verifying model components...")
    model = converter._model
    components = {}

    checks = {
        "model": lambda m: m.get("model") is not None,
        "vocoder": lambda m: m.get("vocoder") is not None,
        "semantic_encoder": lambda m: m.get("semantic_fn") is not None,
        "campplus": lambda m: m.get("campplus") is not None,
        "f0_extractor": lambda m: m.get("f0_extractor") is not None,
        "mel_fn": lambda m: m.get("mel_fn") is not None,
    }

    all_pass = True
    for name, check in checks.items():
        passed = check(model) if model else False
        components[name] = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:20s} {components[name]}")

    result["components"] = components
    result["components_pass"] = all_pass

    if not all_pass:
        print("\n  WARNING: Some components failed verification.")
        result["success"] = False
    else:
        result["success"] = True
        print("\n  All components PASS.")

    # ── Step 6: Unload ──────────────────────────────────────────────
    print("\n[6] Unloading model...")
    try:
        converter.unload()
        result["unload"] = "PASS"
        print("  Unload PASS")
    except Exception as e:
        result["unload"] = f"FAIL: {e}"
        print(f"  Unload FAIL: {e}")

    # ── Step 7: Force garbage collection ────────────────────────────
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    _print_report(result)

    # Save report
    os.makedirs("results", exist_ok=True)
    report_path = "results/model_load_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    return 0 if result["success"] else 1


def _print_report(result):
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)

    sections = [
        ("Preset", result.get("preset")),
        ("Device", result.get("device")),
        ("Sample rate", f"{result.get('sample_rate')} Hz" if result.get('sample_rate') else None),
        ("Seed-VC path", result.get("seedvc_path")),
        ("Config path", result.get("config_path")),
        ("Checkpoint", result.get("checkpoint")),
    ]
    for label, value in sections:
        if value:
            print(f"  {label}: {value}")

    mem_section = [
        ("RAM before", f"{result.get('ram_before_mb')} MB"),
        ("RAM after", f"{result.get('ram_after_mb')} MB"),
        ("System RAM", f"{result.get('system_ram_before_gb')} / {result.get('system_ram_total_gb')} GB"),
        ("Load duration", f"{result.get('load_duration_seconds')}s"),
    ]
    print("\n  Memory & Timing:")
    for label, value in mem_section:
        if value and result.get(label.lower().replace(" ", "_").replace("ram_before", "ram_before_mb").replace("ram_after", "ram_after_mb").replace("system_ram", "system_ram_before_gb").replace("load_duration", "load_duration_seconds")):
            print(f"    {label}: {value}")

    # Simpler approach
    print(f"\n  RAM before:       {result.get('ram_before_mb')} MB")
    print(f"  RAM after:        {result.get('ram_after_mb')} MB")
    print(f"  System RAM:       {result.get('system_ram_before_gb')} / {result.get('system_ram_total_gb')} GB")
    print(f"  Load duration:    {result.get('load_duration_seconds')}s")

    if result.get("gpu_memory_after", {}).get("available"):
        g = result["gpu_memory_after"]
        print(f"  GPU allocated:    {g.get('allocated_mb', 0):.1f} MB")
    if result.get("mps_memory_after", {}).get("available"):
        m = result["mps_memory_after"]
        print(f"  MPS allocated:    {m.get('allocated_mb', 0):.1f} MB")

    print("\n  Components:")
    for name, status in result.get("components", {}).items():
        print(f"    {name:20s} {status}")

    print(f"\n  Unload:           {result.get('unload')}")
    print(f"\n  Overall:          {'PASS' if result.get('success') else 'FAIL'}")

    if result.get("error"):
        print(f"\n  Error stage:      {result['error'].get('stage')}")
        print(f"  Error type:       {result['error'].get('type', 'N/A')}")
        print(f"  Error message:    {result['error'].get('message', 'N/A')}")


if __name__ == "__main__":
    sys.exit(main())
