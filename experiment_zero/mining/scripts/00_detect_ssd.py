#!/usr/bin/env python3
"""Phase 0: Detect and verify AYUSH_SSD mount."""
from pathlib import Path
import subprocess
import json
import os

SSD_CANDIDATES = ["/Volumes/AYUSH_SSD", "/Volumes/AYUSH_SSD1", "/Volumes/AYUSH_SSD2"]

def detect_ssd():
    """Find AYUSH_SSD mount point."""
    for candidate in SSD_CANDIDATES:
        p = Path(candidate)
        if p.exists() and p.is_dir() and p != Path("/"):
            # Verify it's actually a volume (has .Trashes or .fseventsd)
            if (p / ".Trashes").exists() or (p / ".fseventsd").exists():
                return p
    # Fallback: search /Volumes
    result = subprocess.run(["ls", "/Volumes/"], capture_output=True, text=True)
    for line in result.stdout.strip().split("\n"):
        if "AYUSH" in line or "SSD" in line:
            return Path(f"/Volumes/{line.strip()}")
    return None

def get_disk_info(path):
    """Get disk capacity info."""
    try:
        result = subprocess.run(["df", "-h", str(path)], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            return {
                "total": parts[1],
                "used": parts[2],
                "free": parts[3],
                "use_percent": parts[4],
                "mount_point": parts[5] if len(parts) > 5 else parts[0]
            }
    except Exception:
        pass
    return {}

if __name__ == "__main__":
    ssd = detect_ssd()
    if not ssd:
        print("FATAL: AYUSH_SSD not found")
        exit(1)
    
    disk_info = get_disk_info(ssd)
    
    print(f"SSD_ROOT: {ssd}")
    print(f"Mount: {disk_info.get('mount_point', 'unknown')}")
    print(f"Total: {disk_info.get('total', 'unknown')}")
    print(f"Used: {disk_info.get('used', 'unknown')}")
    print(f"Free: {disk_info.get('free', 'unknown')}")
    print(f"Use%: {disk_info.get('use_percent', 'unknown')}")
    
    # Write config
    config = {
        "ssd_root": str(ssd),
        "mount_point": disk_info.get('mount_point', str(ssd)),
        "total_bytes": disk_info.get('total', ''),
        "free_bytes": disk_info.get('free', ''),
        "use_percent": disk_info.get('use_percent', ''),
    }
    
    out_path = Path("/Users/ayushmh/accentedge/experiment_zero/mining/mining_config.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig written to {out_path}")
