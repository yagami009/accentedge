#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo " AccentEdge Phase 1 — Full Gate Pipeline"
echo "============================================================"

# ── Pre-flight: verify we're in Colab ─────────────────────────────────────────
if [ ! -d "/content" ]; then
    echo "ERROR: /content not found — this script must be run in Google Colab."
    exit 1
fi

# ── Install system deps ───────────────────────────────────────────────────────
echo ""
echo "[setup] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq espeak-ng ffmpeg 2>&1 | grep -v "^$" | head -5 || true
echo "[setup] System dependencies installed."

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
echo "[setup] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || {
    echo "WARNING: No GPU detected. Training will be slow."
}

# ── Clone repos ───────────────────────────────────────────────────────────────
ACCENTEDGE_DIR="/content/accentedge"
FA_CODEC_DIR="/content/FAcodec"

if [ ! -d "$ACCENTEDGE_DIR/.git" ]; then
    echo ""
    echo "[setup] Cloning accentedge repo..."
    git clone https://github.com/yagami009/accentedge.git "$ACCENTEDGE_DIR"
fi

if [ ! -d "$FA_CODEC_DIR/.git" ]; then
    echo ""
    echo "[setup] Cloning FAcodec repo..."
    git clone --depth 1 https://github.com/Plachta/FAcodec.git "$FA_CODEC_DIR"
fi

# ── Install Python deps ──────────────────────────────────────────────────────
echo ""
echo "[setup] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r "$ACCENTEDGE_DIR/colab/requirements.txt" 2>&1 | tail -3
echo "[setup] Python dependencies installed."

# ── Environment ──────────────────────────────────────────────────────────────
export PYTHONPATH="$FA_CODEC_DIR:$ACCENTEDGE_DIR/src:$PYTHONPATH"

# ── Helper: get git SHA ──────────────────────────────────────────────────────
get_git_sha() {
    cd "$ACCENTEDGE_DIR"
    git rev-parse --short HEAD 2>/dev/null || echo "unknown"
}

GIT_SHA=$(get_git_sha)
DRIVE_BASE="/content/drive/MyDrive/accentedge/runs/${GIT_SHA}"
mkdir -p "$DRIVE_BASE"

echo ""
echo "Run base: $DRIVE_BASE"
echo "Git SHA: $GIT_SHA"
echo ""

# ── Gate 1 ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Running Gate 1 — Reconstruction Equivalence"
echo "============================================================"

if [ -f "$DRIVE_BASE/gate1/PASS" ]; then
    echo "[skip] Gate 1 already passed at $DRIVE_BASE/gate1/"
else
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=600 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output "$DRIVE_BASE/gate1/executed.ipynb" \
        "$ACCENTEDGE_DIR/colab/01_gate1_reconstruction.ipynb" 2>&1

    if [ -f "$DRIVE_BASE/gate1/PASS" ]; then
        echo "Gate 1 PASSED"
    else
        echo "Gate 1 FAILED — aborting pipeline."
        exit 1
    fi
fi

# ── Gate 2 ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Running Gate 2 — Identity Preservation"
echo "============================================================"

if [ -f "$DRIVE_BASE/gate2/PASS" ]; then
    echo "[skip] Gate 2 already passed at $DRIVE_BASE/gate2/"
else
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=600 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output "$DRIVE_BASE/gate2/executed.ipynb" \
        "$ACCENTEDGE_DIR/colab/02_gate2_identity.ipynb" 2>&1

    if [ -f "$DRIVE_BASE/gate2/PASS" ]; then
        echo "Gate 2 PASSED"
    else
        echo "Gate 2 FAILED — aborting pipeline."
        exit 1
    fi
fi

# ── Gate 3 ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Running Gate 3 — Phoneme Pipeline"
echo "============================================================"

if [ -f "$DRIVE_BASE/gate3/PASS" ]; then
    echo "[skip] Gate 3 already passed at $DRIVE_BASE/gate3/"
else
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=900 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output "$DRIVE_BASE/gate3/executed.ipynb" \
        "$ACCENTEDGE_DIR/colab/03_gate3_phoneme_pipeline.ipynb" 2>&1

    if [ -f "$DRIVE_BASE/gate3/PASS" ]; then
        echo "Gate 3 PASSED"
    else
        echo "Gate 3 FAILED — aborting pipeline."
        exit 1
    fi
fi

# ── Gate 5 (overfit) ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Running Gate 5 — Overfit Training"
echo "============================================================"

if [ -f "$DRIVE_BASE/gate5/PASS" ]; then
    echo "[skip] Gate 5 already passed at $DRIVE_BASE/gate5/"
else
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=1800 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output "$DRIVE_BASE/gate5/executed.ipynb" \
        "$ACCENTEDGE_DIR/colab/04_gate5_overfit.ipynb" 2>&1

    if [ -f "$DRIVE_BASE/gate5/PASS" ]; then
        echo "Gate 5 PASSED"
    else
        echo "Gate 5 FAILED — aborting pipeline."
        exit 1
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ALL GATES PASSED"
echo " Artifacts: $DRIVE_BASE"
echo "============================================================"
