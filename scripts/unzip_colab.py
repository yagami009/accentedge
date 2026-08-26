#!/usr/bin/env python3
"""Unzip accentedge repo on Colab and install."""
import os, subprocess, sys

os.chdir("/content")
os.system("unzip -o accentedge.zip 2>&1 | tail -5")

if os.path.exists("/content/accentedge"):
    print("Unzipped OK")
    os.chdir("/content/accentedge")
    r = subprocess.run(["pip", "install", "-e", "."], capture_output=True, text=True)
    print("pip install:", "OK" if r.returncode == 0 else r.stderr[:300])
    print("GPU:", subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip())
else:
    print("ERROR: /content/accentedge not found")
    sys.exit(1)
