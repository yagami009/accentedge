#!/usr/bin/env python3
"""Phase 1: Full filesystem inventory of AYUSH_SSD."""
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load config
with open(Path(__file__).parent.parent / "mining_config.json") as f:
    config = json.load(f)

SSD_ROOT = Path(config["ssd_root"])
OUT_DIR = Path(config["out_dir"])
INVENTORY_DIR = OUT_DIR / "inventory"
INVENTORY_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm", ".mp4", ".mkv"}
METADATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".arrow", ".txt", ".lab", ".trn", ".trans", ".TextGrid", ".xml", ".yaml", ".yml"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar"}
SKIP_DIRS = {".Trashes", ".fseventsd", ".Spotlight-V100", ".cache", "xet"}

audio_rows = []
metadata_rows = []
archive_rows = []
dir_summary = {}
total_files = 0
start_time = time.time()

def classify_file(path):
    """Classify a file by extension."""
    ext = path.suffix.lower()
    file_type = "other"
    if ext in AUDIO_EXTENSIONS:
        file_type = "audio"
    elif ext in METADATA_EXTENSIONS:
        file_type = "metadata"
    elif ext in ARCHIVE_EXTENSIONS:
        file_type = "archive"
    return file_type

def scan_directory(root_dir):
    """Scan a directory and return file records."""
    records = []
    try:
        for entry in os.scandir(root_dir):
            if entry.is_file():
                try:
                    stat = entry.stat()
                    rel = Path(entry.path).relative_to(SSD_ROOT)
                    records.append({
                        "absolute_path": entry.path,
                        "relative_path": str(rel),
                        "filename": entry.name,
                        "extension": Path(entry.name).suffix.lower(),
                        "size_bytes": stat.st_size,
                        "modified_time": stat.st_mtime,
                        "parent_directory": str(rel.parent),
                        "file_type": classify_file(Path(entry.path)),
                    })
                except (OSError, PermissionError):
                    continue
            elif entry.is_dir() and entry.name not in SKIP_DIRS:
                # Recurse
                sub_records = scan_directory(entry.path)
                records.extend(sub_records)
    except (OSError, PermissionError):
        pass
    return records

print("Starting filesystem inventory...")
print(f"SSD_ROOT: {SSD_ROOT}")
print(f"Free space: {config.get('free_gb', '?')} GB")

# Scan top-level directories first to get overview
print("\n=== Top-level directory overview ===")
for item in sorted(SSD_ROOT.iterdir()):
    if item.name in SKIP_DIRS:
        continue
    try:
        if item.is_dir():
            count = sum(1 for _ in item.rglob("*") if _.is_file())
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            print(f"  {item.name}: {count} files, {size/1e9:.2f} GB")
    except (OSError, PermissionError):
        print(f"  {item.name}: (access denied)")

# Now do full inventory
print("\nScanning all files (this may take a few minutes)...")
all_records = scan_directory(str(SSD_ROOT))

print(f"Total files indexed: {len(all_records)}")

# Separate by type
audio_records = [r for r in all_records if r["file_type"] == "audio"]
metadata_records = [r for r in all_records if r["file_type"] == "metadata"]
archive_records = [r for r in all_records if r["file_type"] == "archive"]

print(f"Audio files: {len(audio_records)}")
print(f"Metadata files: {len(metadata_records)}")
print(f"Archive files: {len(archive_records)}")

# Write inventory
if audio_records:
    df = pd.DataFrame(audio_records)
    df.to_parquet(INVENTORY_DIR / "files_audio.parquet", index=False)
    print(f"Audio inventory: {INVENTORY_DIR / 'files_audio.parquet'}")

if metadata_records:
    df = pd.DataFrame(metadata_records)
    df.to_parquet(INVENTORY_DIR / "files_metadata.parquet", index=False)
    print(f"Metadata inventory: {INVENTORY_DIR / 'files_metadata.parquet'}")

if archive_records:
    df = pd.DataFrame(archive_records)
    df.to_parquet(INVENTORY_DIR / "files_archives.parquet", index=False)
    print(f"Archive inventory: {INVENTORY_DIR / 'files_archives.parquet'}")

# Write all files
if all_records:
    df = pd.DataFrame(all_records)
    df.to_parquet(INVENTORY_DIR / "files_all.parquet", index=False)
    print(f"Full inventory: {INVENTORY_DIR / 'files_all.parquet'}")

elapsed = time.time() - start_time
print(f"\nInventory complete in {elapsed:.1f}s")
