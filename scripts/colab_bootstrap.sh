#!/bin/bash
# scripts/colab_bootstrap.sh
# Sets up AccentEdge Phase 1 on Google Colab CUDA.
# Run this FIRST in any Colab session.
set -e

echo "=== AccentEdge Phase 1 — Colab Bootstrap ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "Python: $(python3 --version)"
echo "Date: $(date -u)"

# Install system deps
apt-get update -qq
apt-get install -y -qq espeak-ng libespeak-ng1 ffmpeg > /dev/null 2>&1
echo "espeak-ng installed"

# Create venv
python3 -m venv /content/.venv --without-pip 2>/dev/null || true
source /content/.venv/bin/activate

# Install uv for fast package management
pip install -q uv 2>/dev/null || python3 -m ensurepip

# Clone Amphion (FACodec dependency) if not present
if [ ! -d "/content/Amphion" ]; then
    echo "Cloning Amphion..."
    git clone --depth 1 https://github.com/open-mmlab/Amphion.git /content/Amphion
else
    echo "Amphion already present"
fi

# Clone FAC-FACodec if not present
if [ ! -d "/content/FAC-FACodec" ]; then
    echo "Cloning FAC-FACodec..."
    git clone https://github.com/Claussss/FAC-FACodec.git /content/FAC-FACodec
else
    echo "FAC-FACodec already present"
fi

# Install project
cd /content
if [ ! -d "/content/accentedge" ]; then
    # Clone from git (preferred) or copy from mounted drive
    if [ -n "$ACCENTEDGE_REPO_URL" ]; then
        git clone "$ACCENTEDGE_REPO_URL" /content/accentedge
    elif [ -d "/content/drive/MyDrive/accentedge" ]; then
        cp -r /content/drive/MyDrive/accentedge /content/accentedge
    else
        echo "WARNING: No accentedge repo found. Will create minimal structure."
        mkdir -p /content/accentedge/{src/accentedge/{codec,phase1,evaluation},scripts,configs/phase1,data,tests}
    fi
fi

cd /content/accentedge

# Install deps
uv pip install -e ".[dev]" --python /content/.venv/bin/python 2>/dev/null || {
    echo "Falling back to pip..."
    pip install -e ".[dev]" 2>/dev/null || true
}

# Install Amphion deps
cd /content/Amphion
pip install -q -r requirements.txt 2>/dev/null || true

# Install FAC-FACodec deps
cd /content/FAC-FACodec
pip install -q -r requirements.txt 2>/dev/null || true

# Install FACodec standalone
if [ ! -d "/content/FAcodec" ]; then
    git clone https://github.com/Plachtaa/FAcodec.git /content/FAcodec
fi
cd /content/FAcodec
pip install -q -r requirements.txt 2>/dev/null || true

echo "=== Bootstrap complete ==="
echo "Activate venv: source /content/.venv/bin/activate"
echo "Run verify: python scripts/verify_cuda.py"
