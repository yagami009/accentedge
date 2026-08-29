#!/usr/bin/env python3
"""Background dataset downloader for AccentEdge.
Downloads datasets in phases using HuggingFace datasets library where available.
Logs progress to /Volumes/AYUSH_SSD/accentedge_data/download.log

Phase 0: CMU ARCTIC + L2-ARCTIC (needed now)
Phase 1: LibriTTS-R + VCTK + FLEURS (starts after Phase 0 completes)
Phase 2: IndicVoices + Project Vaani + Common Voice en_IN (needs permission + HF login)
"""

import os
import sys
import time
import signal
import logging
import subprocess
import urllib.request
import tarfile
import zipfile
from pathlib import Path
from datetime import datetime

# ─── Config ───
SSD = Path("/Volumes/AYUSH_SSD/accentedge_data")
LOG = SSD / "download.log"
PHASE0_DIR = SSD / "phase0"
PHASE1_DIR = SSD / "phase1"
PHASE2_DIR = SSD / "phase3"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("downloader")

DATASETS = {
    # Phase 0 — needed now (downloaded via HF datasets library)
    "phase0": [
        {
            "name": "CMU_ARCTIC",
            "hf_dataset": "MikhailT/cmu-arctic",
            "dest_dir": PHASE0_DIR / "cmu_arctic",
            "size_gb": 1.58,
            "method": "hf",
        },
        {
            "name": "L2_ARCTIC",
            "hf_dataset": "KoelLabs/L2Arctic",
            "dest_dir": PHASE0_DIR / "l2-arctic",
            "size_gb": 0.47,
            "method": "hf",
        },
    ],
    # Phase 1 — starts after Phase 0 completes
    "phase1": [
        {
            "name": "LibriTTS-R_train_clean_100",
            "url": "https://www.openslr.org/resources/141/train_clean_100.tar.gz",
            "dest": PHASE1_DIR / "train_clean_100.tar.gz",
            "extract_to": PHASE1_DIR / "train_clean_100",
            "size_gb": 8.1,
            "method": "url",
        },
        {
            "name": "LibriTTS-R_train_clean_360",
            "url": "https://www.openslr.org/resources/141/train_clean_360.tar.gz",
            "dest": PHASE1_DIR / "train_clean_360.tar.gz",
            "extract_to": PHASE1_DIR / "train_clean_360",
            "size_gb": 28.0,
            "method": "url",
        },
        {
            "name": "VCTK",
            "url": "https://huggingface.co/datasets/confit/vctk-full/resolve/main/vctk.zip",
            "dest": PHASE1_DIR / "vctk.zip",
            "extract_to": PHASE1_DIR / "vctk",
            "size_gb": 10.3,
            "method": "url",
        },
        {
            "name": "FLEURS_en_in",
            "hf_dataset": "google/fleurs",
            "hf_config": "en_us",
            "dest_dir": PHASE1_DIR / "fleurs_en_in",
            "size_gb": 0.5,
            "method": "hf",
        },
    ],
    # Phase 2 — needs permission (remove PHASE2_PAUSE file to start)
    "phase2": [
        {
            "name": "IndicVoices_hindi",
            "hf_dataset": "ai4bharat/IndicVoices",
            "hf_config": "hindi",
            "dest_dir": PHASE2_DIR / "indicvoices_hindi",
            "size_gb": 30,
            "method": "hf",
        },
        {
            "name": "IndicVoices_tamil",
            "hf_dataset": "ai4bharat/IndicVoices",
            "hf_config": "tamil",
            "dest_dir": PHASE2_DIR / "indicvoices_tamil",
            "size_gb": 25,
            "method": "hf",
        },
        {
            "name": "IndicVoices_telugu",
            "hf_dataset": "ai4bharat/IndicVoices",
            "hf_config": "telugu",
            "dest_dir": PHASE2_DIR / "indicvoices_telugu",
            "size_gb": 25,
            "method": "hf",
        },
        {
            "name": "Vaani_Bangalore",
            "hf_dataset": "ayushmh/Vaani-bucket",
            "hf_config": "Karnataka_Bangalore",
            "dest_dir": PHASE2_DIR / "vaani_bangalore",
            "size_gb": 25,
            "method": "hf",
        },
        {
            "name": "Vaani_Hyderabad",
            "hf_dataset": "ayushmh/Vaani-bucket",
            "hf_config": "Telangana_Hyderabad",
            "dest_dir": PHASE2_DIR / "vaani_hyderabad",
            "size_gb": 25,
            "method": "hf",
        },
        {
            "name": "Vaani_Chennai",
            "hf_dataset": "ayushmh/Vaani-bucket",
            "hf_config": "TamilNadu_Chennai",
            "dest_dir": PHASE2_DIR / "vaani_chennai",
            "size_gb": 25,
            "method": "hf",
        },
        {
            "name": "Vaani_Mumbai",
            "hf_dataset": "ayushmh/Vaani-bucket",
            "hf_config": "Maharashtra_Mumbaisuburban",
            "dest_dir": PHASE2_DIR / "vaani_mumbai",
            "size_gb": 25,
            "method": "hf",
        },
        {
            "name": "Vaani_Visakhapatnam",
            "hf_dataset": "ayushmh/Vaani-bucket",
            "hf_config": "AndhraPradesh_Vishakapattanam",
            "dest_dir": PHASE2_DIR / "vaani_visakhapatnam",
            "size_gb": 25,
            "method": "hf",
        },
        {
            "name": "Common_Voice_en_IN",
            "hf_dataset": "mozilla-foundation/common_voice_17_0",
            "hf_config": "en_in",
            "dest_dir": PHASE2_DIR / "common_voice_en_in",
            "size_gb": 20,
            "method": "hf",
        },
    ],
}


def download_from_hf(ds):
    """Download dataset using HuggingFace datasets library."""
    name = ds["name"]
    dataset_id = ds["hf_dataset"]
    dest = ds["dest_dir"]
    config = ds.get("hf_config", None)

    log.info(f"[START] Downloading {name} from HF ({dataset_id}) -> {dest}")

    if dest.exists() and any(dest.iterdir()):
        log.info(f"[SKIP] {name} already downloaded")
        return True

    try:
        from datasets import load_dataset

        log.info(f"  Loading dataset (this may take a while)...")
        if config:
            ds_obj = load_dataset(dataset_id, config, trust_remote_code=False)
        else:
            ds_obj = load_dataset(dataset_id, trust_remote_code=False)

        log.info(f"  Saving to {dest}...")
        for split_name, split_data in ds_obj.items():
            split_dir = dest / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_data.save_to_disk(str(split_dir))
            log.info(f"  Saved split '{split_name}': {len(split_data)} samples -> {split_dir}")
        log.info(f"[DONE] {name} downloaded to {dest}")
        return True

    except Exception as e:
        log.info(f"[FAIL] {name}: {e}")
        return False


def download_url(ds):
    """Download file from URL with resume support and size verification."""
    name = ds["name"]
    url = ds["url"]
    dest = Path(ds["dest"])
    extract_to = Path(ds.get("extract_to", ""))
    expected_size = ds.get("size_gb", 0) * 1024**3  # rough expected size in bytes

    # Skip if already extracted
    if extract_to.exists() and any(extract_to.iterdir()):
        log.info(f"[SKIP] {name} already downloaded and extracted")
        return True

    # Skip if already downloaded AND looks complete (within 5% of expected)
    if dest.exists() and dest.stat().st_size > 1024:
        actual_mb = dest.stat().st_size / 1024**2
        if expected_size > 0 and dest.stat().st_size >= expected_size * 0.95:
            log.info(f"[SKIP] {name} already downloaded ({actual_mb:.1f} MB)")
            if extract_to:
                extract_dataset(name, dest, extract_to)
            return True
        elif expected_size > 0:
            log.info(f"[RESUME] {name} partial download ({actual_mb:.1f} MB / ~{expected_size/1024**3:.1f} GB), resuming...")
        else:
            log.info(f"[SKIP] {name} already exists ({actual_mb:.1f} MB)")
            if extract_to:
                extract_dataset(name, dest, extract_to)
            return True

    log.info(f"[START] Downloading {name}")
    log.info(f"  URL: {url}")
    log.info(f"  Dest: {dest}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Use curl with resume support, longer timeout, and follow redirects
        result = subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(dest),
             "--connect-timeout", "30",
             "--max-time", "14400",  # 4 hours per file
             "--retry", "5",
             "--retry-delay", "10",
             "--retry-max-time", "14400",
             url],
            timeout=14400 + 60,  # slightly longer than curl timeout
        )

        if result.returncode != 0:
            log.info(f"[FAIL] {name}: curl exit code {result.returncode}")
            return False

        size_mb = dest.stat().st_size / 1024**2
        size_gb = dest.stat().st_size / 1024**3
        log.info(f"[DONE] {name} downloaded ({size_mb:.1f} MB / {size_gb:.2f} GB)")

        # Verify size if we have an estimate
        if expected_size > 0 and dest.stat().st_size < expected_size * 0.9:
            log.info(f"[WARN] {name} size ({size_gb:.2f} GB) is much smaller than expected (~{expected_size/1024**3:.1f} GB)")
            log.info(f"  Will retry on next run")

        if extract_to:
            extract_dataset(name, dest, extract_to)

        return True

    except subprocess.TimeoutExpired:
        log.info(f"[FAIL] {name}: Download timeout (>4 hours)")
        return False
    except Exception as e:
        log.info(f"[FAIL] {name}: {e}")
        return False


def extract_dataset(name, dest, extract_to):
    """Extract tar.gz or zip file."""
    if extract_to.exists() and any(extract_to.iterdir()):
        log.info(f"[SKIP] {name} already extracted")
        return True

    log.info(f"[START] Extracting {name}")
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(extract_to)
        elif ".tar.gz" in dest.name or ".tgz" in dest.name:
            with tarfile.open(dest, "r:*") as tf:
                tf.extractall(extract_to)
        log.info(f"[DONE] {name} extracted to {extract_to}")
        return True
    except Exception as e:
        log.info(f"[FAIL] Extract {name}: {e}")
        return False


def run_phase(phase_name, datasets):
    """Run one download phase. Stops if any dataset fails."""
    log.info(f"\n{'='*60}")
    log.info(f"PHASE: {phase_name}")
    log.info(f"{'='*60}")

    failures = []
    for ds in datasets:
        if ds.get("skip"):
            log.info(f"[SKIP] {ds['name']}: flagged to skip")
            continue
        if ds.get("method") == "hf":
            ok = download_from_hf(ds)
        else:
            ok = download_url(ds)

        if not ok:
            failures.append(ds["name"])

    if failures:
        log.info(f"\n[WARN] Phase {phase_name} completed with {len(failures)} failures:")
        for f in failures:
            log.info(f"  - {f}")
        log.info("Continuing to next phase anyway...")
    else:
        log.info(f"\nPhase {phase_name} complete — all datasets downloaded successfully")

    return len(failures) == 0


def main():
    log.info(f"Downloader started at {datetime.now().isoformat()}")
    log.info(f"SSD: {SSD}")
    log.info(f"Available space: check with 'df -h {SSD}'")

    # Phase 0 — needed now
    run_phase("0 (Phase 0 — TGFP v2 datasets)", DATASETS["phase0"])

    # Phase 1 — benchmark + training
    run_phase("1 (Phase 1 — Benchmark + Training)", DATASETS["phase1"])

    # Phase 2 — needs explicit permission
    log.info(f"\n{'='*60}")
    log.info(f"Phase 2 (Indian pretraining) requires permission")
    log.info(f"Datasets: {', '.join(d['name'] for d in DATASETS['phase2'])}")
    log.info(f"Estimated size: {sum(d['size_gb'] for d in DATASETS['phase2']):.0f} GB")
    log.info(f"To start Phase 2, remove the PHASE2_PAUSE file:")
    log.info(f"  rm {SSD}/PHASE2_PAUSE")
    log.info(f"{'='*60}")

    pause_file = SSD / "PHASE2_PAUSE"
    pause_file.touch()
    log.info(f"Created pause file: {pause_file}")
    log.info("Waiting for PHASE2_PAUSE to be removed...")

    while pause_file.exists():
        time.sleep(30)

    log.info("PHASE2_PAUSE removed — starting Phase 2")
    run_phase("2 (Phase 2 — Indian pretraining)", DATASETS["phase2"])

    log.info(f"\nAll downloads complete at {datetime.now().isoformat()}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    main()
