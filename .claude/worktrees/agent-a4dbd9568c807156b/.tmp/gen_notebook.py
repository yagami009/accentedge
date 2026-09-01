#!/usr/bin/env python3
"""Generate the Gate 4 Colab notebook JSON."""
import json
import os

nb = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def md(src):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

# Cell 0: Title
md([
    "# Gate 4 - Indian-English Accent Conversion Strength Sweep",
    "",
    "**Purpose:** Verify that AccentConverter produces monotonically increasing accent change at multiple strength levels while preserving acoustic quality and speaker identity.",
    "",
    "**Pass criteria:**",
    "- Identity shift at strength=0 ~= 0",
    "- Identity shift at strength=1 > 0.15",
    "- Identity shift increases monotonically",
    "- mel L1 < 0.5 at all strengths",
    "",
    "**Method:**",
    "1. Load L2-ARCTIC Indian-English test utterances",
    "2. Run AccentConverter at 5 strength levels per utterance",
    "3. Measure mel L1, ECAPA identity shift, and WER",
    "4. Plot strength curves and generate pass/fail table",
    "5. Save artifacts to Drive"
])

# Cell 1: Mount Drive
code([
    "# 1. Mount Google Drive",
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "print('Drive mounted at /content/drive')"
])

# Cell 2: Install system deps
code([
    "# 2. Install system dependencies (espeak-ng for phonemizer)",
    "!apt-get update -qq",
    "!apt-get install -y -qq espeak-ng > /dev/null 2>&1",
    "print('espeak-ng installed')"
])

# Cell 3: Setup paths and clone repos
code([
    "# 3. Setup paths and clone repos",
    "import os, sys, json, time, subprocess, types, warnings, shutil",
    "from pathlib import Path",
    "",
    "warnings.simplefilter('ignore')",
    "",
    "ACCENTEDGE_DIR = '/content/accentedge'",
    "FA_CODEC_DIR = '/content/FAcodec'",
    "GATE_DIR = '/content/gate4_artifacts'",
    "DRIVE_BASE = '/content/drive/MyDrive/accentedge/runs'",
    "",
    "os.makedirs(GATE_DIR, exist_ok=True)",
    "os.makedirs(f'{GATE_DIR}/audio', exist_ok=True)",
    "",
    "SAMPLE_RATE = 24000",
    "",
    "# Clone repos if not present",
    "def run(cmd, desc=''):",
    "    print(f'>>> {desc or cmd[:80]}')",
    "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)",
    "    out = r.stdout.strip()",
    "    if out:",
    "        print(out[:500])",
    "    return r",
    "",
    "if not os.path.exists(ACCENTEDGE_DIR):",
    "    run('git clone https://github.com/user/accentedge.git /content/accentedge', 'Clone AccentEdge')",
    "else:",
    "    print('AccentEdge repo exists.')",
    "",
    "if not os.path.exists(FA_CODEC_DIR):",
    "    run('git clone https://github.com/Plachta/FAcodec.git /content/FAcodec', 'Clone FAcodec')",
    "else:",
    "    print('FAcodec repo exists.')",
    "",
    "os.chdir(ACCENTEDGE_DIR)"
])

# Cell 4: Install pip deps
code([
    "# 4. Install Python dependencies",
    "!pip install -q torch torchaudio transformers speechbrain faster-whisper phonemizer",
    "!pip install -q librosa jiwer soundfile numpy scipy pyyaml einops huggingface-hub matplotlib",
    "print('Dependencies installed')"
])

# Cell 5: Setup PYTHONPATH and mocks
code([
    "# 5. Setup environment",
    "sys.path.insert(0, f'{ACCENTEDGE_DIR}/src')",
    "sys.path.insert(0, FA_CODEC_DIR)",
    "",
    "# Mock audiotools before FAcodec imports (same pattern as gate scripts)",
    "def _make_mock(name):",
    "    m = types.ModuleType(name)",
    "    m.__path__ = []",
    "    m.__package__ = name",
    "    return m",
    "",
    "mock_audio = _make_mock('audiotools')",
    "mock_ml = _make_mock('audiotools.ml')",
    "mock_ml.BaseModel = type('BaseModel', (), {'INTERN': [], 'EXTERN': []})",
    "mock_audio.ml = mock_ml",
    "mock_audio.AudioSignal = type('AudioSignal', (), {})",
    "mock_audio.STFTParams = type('STFTParams', (), {})",
    "mock_core = _make_mock('audiotools.core')",
    "mock_core.util = _make_mock('audiotools.core.util')",
    "sys.modules['audiotools'] = mock_audio",
    "sys.modules['audiotools.ml'] = mock_ml",
    "sys.modules['audiotools.core'] = mock_core",
    "sys.modules['audiotools.core.util'] = mock_core.util",
    "",
    "print('Environment configured')"
])

# Cell 6: Import and run gate4
code([
    "# 6. Run Gate 4 strength sweep",
    "import torch",
    "import numpy as np",
    "",
    "from scripts.gate4_strength_sweep import Gate4Config, _load_l2_arctic_samples, _build_converter, run_strength_sweep",
    "",
    "cfg = Gate4Config()",
    "cfg.output_dir = Path(GATE_DIR)",
    "cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'",
    "cfg.n_samples = 5",
    "cfg.strengths = [0.0, 0.25, 0.5, 0.75, 1.0]",
    "print(f'Device: {cfg.device}')",
    "print(f'Strengths: {cfg.strengths}')",
    "print(f'Samples: {cfg.n_samples}')",
    "",
    "# Load L2-ARCTIC Hindi samples",
    "samples = _load_l2_arctic_samples(cfg.n_samples, cfg)",
    "print(f'Loaded {len(samples)} samples')",
    "for i, s in enumerate(samples):",
    "    print(f'  [{i}] {s[\"speaker\"]} | {s[\"wav_path\"]} | {s[\"duration\"]:.1f}s | {s[\"transcript\"][:50]}...')",
    "",
    "# Build converter",
    "converter = _build_converter(cfg, torch.device(cfg.device))",
    "",
    "# Run sweep",
    "result = run_strength_sweep(samples, converter, cfg)"
])

# Cell 7: Plot identity shift curves
code([
    "# 7. Plot identity shift vs strength",
    "import matplotlib.pyplot as plt",
    "import json",
    "",
    "with open(f'{GATE_DIR}/strength_curves.json') as f:",
    "    curves = json.load(f)",
    "",
    "strengths = [0.0, 0.25, 0.5, 0.75, 1.0]",
    "fig, ax = plt.subplots(figsize=(10, 6))",
    "",
    "# Plot per-sample lines",
    "samples = result['per_sample']",
    "for sid, sdata in enumerate(samples):",
    "    shifts = [sdata['strengths'][str(s)]['identity_shift'] for s in strengths]",
    "    ax.plot(strengths, shifts, 'o-', alpha=0.5, label=f'Sample {sid}')",
    "",
    "# Plot mean line",
    "mean_shifts = [curves[str(s)]['identity_shift_mean'] for s in strengths]",
    "ax.plot(strengths, mean_shifts, 'k-o', linewidth=3, markersize=10, label='Mean')",
    "",
    "ax.axhline(y=0.15, color='r', linestyle='--', label='Threshold (0.15)')",
    "ax.axhline(y=0.0, color='g', linestyle='--', label='Zero')",
    "ax.set_xlabel('Conversion Strength', fontsize=12)",
    "ax.set_ylabel('Identity Shift (cosine distance)', fontsize=12)",
    "ax.set_title('Identity Shift vs Strength', fontsize=14)",
    "ax.legend(loc='upper left')",
    "ax.grid(True, alpha=0.3)",
    "ax.set_ylim(-0.05, 0.35)",
    "",
    "plt.tight_layout()",
    "plt.savefig(f'{GATE_DIR}/identity_shift_curves.png', dpi=150)",
    "plt.show()",
    "print('Saved identity_shift_curves.png')"
])

# Cell 8: Plot mel L1 curves
code([
    "# 8. Plot mel L1 vs strength",
    "fig, ax = plt.subplots(figsize=(10, 6))",
    "",
    "for sid, sdata in enumerate(samples):",
    "    mels = [sdata['strengths'][str(s)]['mel_l1'] for s in strengths]",
    "    ax.plot(strengths, mels, 's-', alpha=0.5, label=f'Sample {sid}')",
    "",
    "mean_mels = [curves[str(s)]['mel_l1_mean'] for s in strengths]",
    "ax.plot(strengths, mean_mels, 'k-s', linewidth=3, markersize=10, label='Mean')",
    "",
    "ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')",
    "ax.set_xlabel('Conversion Strength', fontsize=12)",
    "ax.set_ylabel('mel L1', fontsize=12)",
    "ax.set_title('Acoustic Quality: mel L1 vs Strength', fontsize=14)",
    "ax.legend(loc='upper left')",
    "ax.grid(True, alpha=0.3)",
    "ax.set_ylim(0, 1.0)",
    "",
    "plt.tight_layout()",
    "plt.savefig(f'{GATE_DIR}/mel_l1_curves.png', dpi=150)",
    "plt.show()",
    "print('Saved mel_l1_curves.png')"
])

# Cell 9: Plot WER vs strength
code([
    "# 9. Plot WER vs strength (if available)",
    "fig, ax = plt.subplots(figsize=(10, 6))",
    "has_wer = False",
    "",
    "for sid, sdata in enumerate(samples):",
    "    wers = []",
    "    for s in strengths:",
    "        w = sdata['strengths'][str(s)].get('wer')",
    "        if w is not None:",
    "            wers.append(w)",
    "        else:",
    "            wers.append(None)",
    "    if any(w is not None for w in wers):",
    "        has_wer = True",
    "        ax.plot(strengths, wers, 'D-', alpha=0.5, label=f'Sample {sid}')",
    "",
    "if has_wer:",
    "    mean_wers = []",
    "    for s in strengths:",
    "        ws = [sdata['strengths'][str(s)].get('wer') for sdata in samples]",
    "        ws = [w for w in ws if w is not None]",
    "        mean_wers.append(sum(ws)/len(ws) if ws else None)",
    "    ax.plot(strengths, mean_wers, 'k-D', linewidth=3, markersize=10, label='Mean')",
    "    ax.set_xlabel('Conversion Strength', fontsize=12)",
    "    ax.set_ylabel('Word Error Rate', fontsize=12)",
    "    ax.set_title('Content Preservation: WER vs Strength', fontsize=14)",
    "    ax.legend(loc='upper left')",
    "    ax.grid(True, alpha=0.3)",
    "    plt.tight_layout()",
    "    plt.savefig(f'{GATE_DIR}/wer_curves.png', dpi=150)",
    "    plt.show()",
    "    print('Saved wer_curves.png')",
    "else:",
    "    print('WER data not available (faster-whisper not installed)')",
    "    ax.text(0.5, 0.5, 'WER data not available', transform=ax.transAxes, ha='center', fontsize=14)",
    "    ax.set_title('Content Preservation: WER vs Strength', fontsize=14)",
    "    plt.tight_layout()",
    "    plt.show()"
])

# Cell 10: Audio playback
code([
    "# 10. Audio playback for original and converted at key strengths",
    "import torchaudio",
    "from IPython.display import Audio, display, HTML",
    "import pandas as pd",
    "",
    "key_strengths = [0.0, 0.5, 1.0]",
    "",
    "display(HTML('<h3>Audio Playback: Original vs Converted</h3>'))",
    "",
    "for sid, sdata in enumerate(samples):",
    "    display(HTML(f'<h4>Sample {sid}: {sdata[\"speaker\"]}</h4>'))",
    "    source_wav = np.array(sdata['source_wav'], dtype=np.float32)",
    "    display(Audio(source_wav, rate=SAMPLE_RATE))",
    "    display(HTML('<b>Original</b>'))",
    "    ",
    "    for s in key_strengths:",
    "        cw = np.array(sdata['strengths'][str(s)]['converted_wav'], dtype=np.float32)",
    "        display(HTML(f'<b>Strength={s}</b>'))",
    "        display(Audio(cw, rate=SAMPLE_RATE))",
    "    display(HTML('<hr>'))"
])

# Cell 11: Pass/fail table
code([
    "# 11. Gate 4 pass/fail table",
    "criteria = result['verdict']",
    "",
    "table_data = [",
    "    ['Criterion', 'Value', 'Threshold', 'Pass?'],",
    "    ['Identity shift at strength=0', f\"{curves['0.0']['identity_shift_mean']:.4f}\", '<= 0.05', 'PASS' if criteria['identity_at_0_pass'] else 'FAIL'],",
    "    ['Identity shift at strength=1', f\"{curves['1.0']['identity_shift_mean']:.4f}\", '>= 0.15', 'PASS' if criteria['identity_at_1_pass'] else 'FAIL'],",
    "    ['Monotonically increasing', 'Yes' if criteria['monotonic_pass'] else 'No', 'True', 'PASS' if criteria['monotonic_pass'] else 'FAIL'],",
    "    ['Max mel L1', f\"{max(curves[str(s)]['mel_l1_mean'] for s in [0.0,0.25,0.5,0.75,1.0]):.4f}\", '<= 0.5', 'PASS' if criteria['mel_l1_pass'] else 'FAIL'],",
    "    ['OVERALL GATE 4', '', '', 'PASS' if criteria['gate4_pass'] else 'FAIL'],",
    "]",
    "",
    "df = pd.DataFrame(table_data[1:], columns=table_data[0])",
    "display(HTML('<h3>Gate 4 Results</h3>'))",
    "display(df.style.applymap(lambda x: 'color: green; font-weight: bold' if x == 'PASS' else 'color: red; font-weight: bold', subset=['Pass?']))"
])

# Cell 12: Save to Drive
code([
    "# 12. Save artifacts to Drive",
    "import shutil",
    "from datetime import datetime",
    "",
    "git_sha = subprocess.run(",
    "    ['git', 'rev-parse', '--short', 'HEAD'],",
    "    capture_output=True, text=True",
    ").stdout.strip() or 'nogit'",
    "ts = datetime.now().strftime('%Y%m%d_%H%M%S')",
    "drive_out = f'{DRIVE_BASE}/{git_sha}/gate4_{ts}'",
    "os.makedirs(drive_out, exist_ok=True)",
    "",
    "for fname in os.listdir(GATE_DIR):",
    "    src = f'{GATE_DIR}/{fname}'",
    "    if os.path.isfile(src):",
    "        shutil.copy2(src, f'{drive_out}/{fname}')",
    "        print(f'  Saved: {drive_out}/{fname}')",
    "",
    "print(f'\\nAll artifacts saved to: {drive_out}')",
    "print(f'Gate 4: {\"PASSED\" if criteria[\"gate4_pass\"] else \"FAILED\"}')"
])

# Write notebook
out_path = "/Users/ayushmh/accentedge/colab/04_gate4_strength_sweep.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote notebook with {len(nb['cells'])} cells to {out_path}")
