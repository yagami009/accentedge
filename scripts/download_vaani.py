#!/usr/bin/env python3
"""
Vaani Bucket Downloader
Downloads parquet shards from the Vaani HF bucket for specified districts.

Usage:
    python download_vaani.py                          # Download all 5 districts
    python download_vaani.py --district Bangalore      # Download just one district
    python download_vaani.py --dry-run                 # List files without downloading
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi


# ── Configuration ──────────────────────────────────────────────────────────────

BUCKET_ID = "ayushmh/Vaani-bucket"

DISTRICTS = {
    "Bangalore":           "audio/Karnataka/Bangalore/",
    "Hyderabad":           "audio/Telangana/Hyderabad/",
    "Chennai":             "audio/TamilNadu/Chennai/",
    "Mumbaisuburban":      "audio/Maharashtra/Mumbaisuburban/",
    "Vishakapattanam":     "audio/AndhraPradesh/Vishakapattanam/",
}

DEFAULT_OUTPUT_ROOT = "/Volumes/AYUSH_SSD/accentedge_data/phase3"

TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_token() -> Optional[str]:
    """Read the HF token from the cache file."""
    if TOKEN_PATH.exists():
        return TOKEN_PATH.read_text().strip()
    print(f"⚠  No token found at {TOKEN_PATH}")
    print("   Set HF_TOKEN env var or place your token at that path.")
    return None


def list_bucket_files(api: HfApi, prefix: str) -> list:
    """Return all BucketFile objects under a bucket prefix."""
    return [
        item for item in api.list_bucket_tree(BUCKET_ID, prefix=prefix)
        if item.type == "file"
    ]


def local_path_for_file(output_root: Path, district_key: str, remote_path: str) -> Path:
    """Map a remote path like 'audio/Karnataka/Bangalore/train-00000-of-00069.parquet'
    to a local path like '<root>/vaani_Bangalore/train-00000-of-00069.parquet'."""
    filename = Path(remote_path).name
    return output_root / f"vaani_{district_key}" / filename


def format_size(bytes_val: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(bytes_val) < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


# ── Download ───────────────────────────────────────────────────────────────────

def download_district(
    api: HfApi,
    district_key: str,
    prefix: str,
    output_root: Path,
    token: str,
    dry_run: bool = False,
) -> None:
    """Download all files for one district with resume support."""

    print(f"\n{'─' * 60}")
    print(f"District: {district_key}")
    print(f"Prefix:   {prefix}")
    print(f"Output:   {output_root / f'vaani_{district_key}'}")
    print(f"{'─' * 60}")

    # 1. List files
    print("  Listing files from bucket …")
    files = list_bucket_files(api, prefix)

    if not files:
        print("  ⚠  No files found. Check the prefix.")
        return

    total_bytes = sum(f.size for f in files)
    print(f"  Found {len(files)} files, {format_size(total_bytes)} total")

    # 2. Determine which files need downloading (resume support)
    dest_dir = output_root / f"vaani_{district_key}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    to_download = []
    skipped = 0

    for bf in files:
        local = local_path_for_file(output_root, district_key, bf.path)

        if local.exists() and local.stat().st_size == bf.size:
            skipped += 1
            continue

        to_download.append((bf.path, local))

    if skipped:
        print(f"  ✓  Skipped {skipped} already-downloaded files (resume)")

    if not to_download:
        print("  ✓  All files already present. Nothing to do.")
        return

    remaining_bytes = sum(
        next(f.size for f in files if f.path == bf_path)
        for bf_path, _ in to_download
    )
    print(f"  →  Downloading {len(to_download)} files ({format_size(remaining_bytes)})")

    # 3. Download in batches
    batch_size = 10
    downloaded_bytes = 0
    start_time = time.time()

    for i in range(0, len(to_download), batch_size):
        batch = to_download[i : i + batch_size]
        pairs = [(remote, str(local)) for remote, local in batch]

        if dry_run:
            print(f"  [dry-run] Would download: {[p for p, _ in pairs]}")
            continue

        try:
            api.download_bucket_files(
                bucket_id=BUCKET_ID,
                files=pairs,
                raise_on_missing_files=True,
                token=token,
            )
        except Exception as e:
            print(f"\n  ✗  Error downloading batch: {e}")
            print("  Retrying individual files …")
            for remote, local in pairs:
                try:
                    api.download_bucket_files(
                        bucket_id=BUCKET_ID,
                        files=[(remote, str(local))],
                        raise_on_missing_files=True,
                        token=token,
                    )
                    downloaded_bytes += next(f.size for f in files if f.path == remote)
                except Exception as e2:
                    print(f"    ✗  Failed: {remote} → {e2}")

        # Progress
        batch_completed = sum(
            next(f.size for f in files if f.path == rp)
            for rp, _ in pairs
        )
        downloaded_bytes += batch_completed
        elapsed = time.time() - start_time
        pct = downloaded_bytes / remaining_bytes * 100
        speed = downloaded_bytes / elapsed if elapsed > 0 else 0
        eta = (remaining_bytes - downloaded_bytes) / speed if speed > 0 else 0

        file_idx = min(i + batch_size, len(to_download))
        print(
            f"  [{file_idx}/{len(to_download)} files] "
            f"{pct:.1f}%  "
            f"{format_size(downloaded_bytes)}/{format_size(remaining_bytes)}  "
            f"ETA {int(eta // 60)}m{int(eta % 60)}s"
        )

    elapsed_total = time.time() - start_time
    print(f"\n  ✓  Download complete in {elapsed_total / 60:.1f} min")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Vaani HF bucket data")
    parser.add_argument(
        "--district",
        choices=list(DISTRICTS.keys()),
        help="Download a single district (default: all 5)",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without downloading",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (default: read from ~/.cache/huggingface/token)",
    )
    args = parser.parse_args()

    output_root = Path(args.output)
    token = args.token or load_token()

    if not token:
        print("✗  No HF token available. Aborting.")
        sys.exit(1)

    # Verify SSD mount
    if not output_root.exists():
        print(f"✗  Output directory does not exist: {output_root}")
        sys.exit(1)

    api = HfApi(token=token)

    # Verify auth
    try:
        user = api.whoami()
        print(f"Authenticated as: {user.get('name', user.get('login', 'unknown'))}")
    except Exception as e:
        print(f"✗  Auth failed: {e}")
        sys.exit(1)

    # Select districts
    if args.district:
        selected = {args.district: DISTRICTS[args.district]}
    else:
        selected = DISTRICTS

    # Print summary
    grand_total = 0
    print("\nVaani Bucket Downloader")
    print("=" * 60)
    for key, prefix in selected.items():
        files = list_bucket_files(api, prefix)
        size = sum(f.size for f in files)
        grand_total += size
        print(f"  {key:20s}  {len(files):3d} shards  {format_size(size)}")
    print(f"  {'TOTAL':20s}  {format_size(grand_total)}")

    if args.dry_run:
        print("\n(dry-run mode — no files will be downloaded)")
        return

    # Download each district
    for key, prefix in selected.items():
        try:
            download_district(api, key, prefix, output_root, token)
        except KeyboardInterrupt:
            print("\n\n⚠  Interrupted. Resume is safe — already-downloaded files will be skipped.")
            sys.exit(0)

    # Final summary
    print(f"\n{'=' * 60}")
    print("All downloads complete!")
    print(f"Data location: {output_root}/vaani_<district>/")


if __name__ == "__main__":
    main()
