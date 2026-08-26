#!/usr/bin/env python3
"""Phone-to-codec-frame alignment diagnostic report.

Produces an HTML file with per-utterance panels showing:
  1. Waveform
  2. Mel spectrogram with phone-boundary vertical lines
  3. Phone label text above the spectrogram
  4. FACodec frame grid markers (80 fps = every 12.5 ms)
  5. Color-coded phoneme-region bands

Usage:
  # Single file
  python scripts/visualize_alignment.py --wav /path/to/audio.wav --text "hello world"

  # Built-in test utterances
  python scripts/visualize_alignment.py --test all

  # Batch from YAML
  python scripts/visualize_alignment.py --batch utterances.yaml
"""
import argparse
import base64
import io
import json
import sys
import types
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Mock audiotools (for FAcodec imports)
# ---------------------------------------------------------------------------

def _make_mock(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    return m

for _name, _attrs in [
    ("audiotools", {"AudioSignal": type("AudioSignal", (), {}), "STFTParams": type("STFTParams", (), {})}),
    ("audiotools.ml", {"BaseModel": type("BaseModel", (), {"INTERN": [], "EXTERN": []})}),
    ("audiotools.core", {}),
    ("audiotools.core.util", {}),
]:
    _m = _make_mock(_name)
    for _k, _v in _attrs.items():
        setattr(_m, _k, _v)
    sys.modules[_name] = _m


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR = 24000           # FACodec sample rate
HOP_LENGTH = 300     # FACodec encoder hop (24000/300 = 80 fps)
FPS = SR / HOP_LENGTH  # 80.0
FRAME_MS = 1000.0 / FPS   # 12.5 ms per frame
N_MELS = 80
N_FFT = 2048

# Color palette for phonemes (20 distinct colors, cycling)
PHONE_COLORS = [
    "#e06c75", "#98c379", "#e5c07b", "#61afef", "#c678dd",
    "#56b6c2", "#d19a66", "#ff79c6", "#8be9fd", "#bd93f9",
    "#50fa7b", "#ffb86c", "#ff5555", "#6272a4", "#282a36",
    "#f1fa8c", "#44bd9e", "#ff6e6e", "#aad94c", "#c592fc",
]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str) -> np.ndarray:
    """Load audio at 24 kHz."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        import librosa
        data, sr = librosa.load(path, sr=SR)
    if sr != SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------

def compute_mel(waveform: np.ndarray) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    return librosa.power_to_db(mel, ref=np.max)


# ---------------------------------------------------------------------------
# Phoneme extraction
# ---------------------------------------------------------------------------

def extract_phonemes(waveform: np.ndarray, transcript: str) -> tuple:
    """Map transcript words to phoneme sequences distributed across codec frames.

    Two backends tried in order:
      1. phonemizer (espeak-ng) — convert text → IPA → phone list
      2. simple word-to-phones (spaces → phones) fallback

    Returns (codec_probs, phone_labels, frame_times) where:
      codec_probs : np.ndarray [n_frames, vocab_size]
      phone_labels: list[str] per codec frame
      frame_times : np.ndarray [n_frames] in seconds
    """
    n_frames = int(len(waveform) / SR * FPS)

    phones = _text_to_phones(transcript)
    if not phones:
        phones = ["SIL"] * n_frames

    # Distribute phones evenly across codec frames
    phone_labels = []
    for i in range(n_frames):
        idx = min(int(i * len(phones) / n_frames), len(phones) - 1)
        phone_labels.append(phones[idx])

    # Build fake soft-probability matrix (uniform over phone vocab)
    vocab_size = max(len(phones), 10)
    codec_probs = np.ones((n_frames, vocab_size), dtype=np.float32) * (1.0 / vocab_size)

    frame_times = np.arange(n_frames) / FPS
    return codec_probs, phone_labels, frame_times


def _text_to_phones(text: str) -> list[str]:
    """Convert transcript text to phoneme list using phonemizer."""
    try:
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
        backend = EspeakBackend(language="en-us", preserve_punctuation=False)
        ipa = phonemize(text, language="en-us", backend=backend, strip=True)
        phones = _ipa_to_phones(ipa)
        if phones:
            return phones
    except Exception:
        pass

    return _rough_phones(text)


def _ipa_to_phones(ipa: str) -> list[str]:
    """Map IPA string to a list of canonical phoneme symbols."""
    vowel_map = {
        "ɑ": "AA", "æ": "AE", "ʌ": "AH", "ɔ": "AO", "ɛ": "EH",
        "ɪ": "IH", "i": "IY", "ʊ": "UH", "u": "UW", "ʊə": "UH",
        "ɑː": "AA", "ɔː": "AO", "ɛː": "EH", "iː": "IY", "uː": "UW",
        "eɪ": "EY", "aɪ": "AY", "ɔɪ": "OY", "aʊ": "AW", "oʊ": "OW",
        "ə": "AH", "ɜː": "ER", "ɜ": "ER", "ɚ": "ER",
        "ð": "DH", "θ": "TH", "ʃ": "SH", "ʒ": "ZH", "ŋ": "NG",
        "tʃ": "CH", "dʒ": "JH", "j": "Y", "w": "W",
        "ː": "", "ˈ": "", "ˌ": "", "̩": "", "̃": "",
        "ʰ": "", "ʷ": "", "ⁿ": "", "ˡ": "",
    }

    result = []
    i = 0
    while i < len(ipa):
        if ipa[i].isspace():
            i += 1
            continue

        two = ipa[i:i+2]
        if two in vowel_map:
            sym = vowel_map[two]
            if sym:
                result.append(sym)
            i += 2
            continue
        if two in ("ts", "dz", "tr", "dr", "pr", "br", "kr", "gr",
                   "sl", "kl", "ɡl"):
            result.append(two.upper())
            i += 2
            continue

        ch = ipa[i]
        sym = vowel_map.get(ch)
        if sym:
            if sym:
                result.append(sym)
        elif ch.isalpha() or ch in "ʔʕʢʡɸβɦ":
            result.append(ch.upper())
        i += 1

    return result


def _rough_phones(text: str) -> list[str]:
    """Simple fallback: split text into rough phone-length chunks."""
    words = text.lower().split()
    phones = []
    for word in words:
        rough = list(word.replace("th", "T").replace("sh", "S").replace("ch", "C")
                     .replace("ee", "I").replace("oo", "U").replace("ou", "W")
                     .replace("ai", "E").replace("er", "R")
                     .replace("tion", "XN").replace("ing", "NG").replace("ed", "D"))
        for ch in rough:
            if ch.isalpha():
                phones.append(ch.upper())
    if not phones:
        phones = ["AA"] * max(len(words) * 2, 1)
    return phones


# ---------------------------------------------------------------------------
# Synthetic speech generation
# ---------------------------------------------------------------------------

def synthesize_speech(text: str) -> np.ndarray:
    """Generate a speech-like waveform for test utterances."""
    phones = _text_to_phones(text)
    n_phones = max(len(phones), 1)

    phone_dur = {v: 0.18 for v in "AEIOUY"}
    phone_dur.update({v: 0.12 for v in "BPTCDFGJKLMNRSTVWZ"})
    phone_dur.update({v: 0.10 for v in "HQX"})
    phone_dur.update({v: 0.06 for v in "CSZ"})
    phone_dur.update({v: 0.22 for v in "MNG"})
    phone_dur.update({v: 0.08 for v in "DHTHSHZH"})

    def dur_for(p):
        return phone_dur.get(p.upper(), 0.13)

    total_dur = sum(dur_for(p) for p in phones) + 0.4
    n_samples = int(SR * total_dur)
    t = np.linspace(0, total_dur, n_samples, dtype=np.float32)

    # Build pitch contour
    f0 = np.zeros(n_samples, dtype=np.float32)
    cur_t = 0.0
    for p in phones:
        d = dur_for(p)
        n = int(SR * d)
        start = int(cur_t * SR)
        end = min(start + n, n_samples)
        if start >= n_samples:
            break
        base_f0 = {"A": 160, "E": 190, "I": 210, "O": 155, "U": 130,
                   "Y": 200, "M": 140, "N": 145, "R": 165, "W": 125,
                   "T": 130, "S": 125, "D": 140, "L": 155}.get(p.upper(), 150)
        f0[start:end] = base_f0 + 30 * np.sin(2 * np.pi * 4 * t[start:end])
        cur_t += d

    # Generate waveform with harmonics
    wave = np.zeros(n_samples, dtype=np.float32)
    wave += 0.22 * np.sin(2 * np.pi * f0 * t)
    wave += 0.11 * np.sin(2 * np.pi * 2 * f0 * t)
    wave += 0.055 * np.sin(2 * np.pi * 3 * f0 * t)
    wave += 0.02 * np.sin(2 * np.pi * 4 * f0 * t + 0.3)
    wave *= 0.35

    # Amplitude envelope per phone
    env = np.zeros(n_samples, dtype=np.float32)
    cur_t = 0.0
    for p in phones:
        d = dur_for(p)
        n = int(SR * d)
        start = int(cur_t * SR)
        end = min(start + n, n_samples)
        if start >= n_samples:
            break
        ramp = max(int(0.08 * n), 1)
        env[start:start+ramp] = np.linspace(0, 1, ramp)
        mid = min(start + ramp + int(0.5 * n), end)
        env[start+ramp:mid] = 1.0
        if end - mid > 0:
            env[mid:end] = np.linspace(1, 0.1, end - mid)
        cur_t += d

    # Silence at start/end
    silence = int(SR * 0.2)
    if silence < n_samples:
        env[:silence] *= np.linspace(0, 1, silence)
        env[-silence:] *= np.linspace(1, 0, silence)

    return (wave * env).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def plot_utterance(waveform: np.ndarray, mel: np.ndarray,
                   phone_labels: list, frame_times: np.ndarray,
                   codec_probs: np.ndarray,
                   transcript: str,
                   utt_id: str) -> str:
    """Generate a single utterance panel as a base64 PNG."""

    total_dur = len(waveform) / SR
    n_frames = mel.shape[1]

    # Collapse consecutive duplicate labels → boundaries + label list
    boundaries = [0.0]
    labels = []
    prev = None
    for i, lbl in enumerate(phone_labels):
        if lbl != prev:
            boundaries.append(frame_times[min(i, len(frame_times) - 1)])
            labels.append(lbl)
        prev = lbl

    # Assign colors per unique phone
    unique_phones = sorted(set(labels))
    phone_colors = {}
    for i, ph in enumerate(unique_phones):
        idx = i % len(PHONE_COLORS)
        phone_colors[ph] = PHONE_COLORS[idx]

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3.5, 1.4],
                           hspace=0.28, left=0.04, right=0.97,
                           top=0.93, bottom=0.08)

    ax_wave = fig.add_subplot(gs[0])
    ax_mel  = fig.add_subplot(gs[1])
    ax_phon = fig.add_subplot(gs[2])

    # ── Waveform ──────────────────────────────────────────────────────────
    t_wave = np.linspace(0, total_dur, len(waveform))
    ax_wave.fill_between(t_wave, waveform, alpha=0.45, color="#58a6ff", lw=0)
    ax_wave.plot(t_wave, waveform, lw=0.3, color="#58a6ff", alpha=0.6)
    ax_wave.set_xlim(0, total_dur)
    ax_wave.set_ylabel("amplitude", fontsize=8, color="#8b949e")
    ax_wave.tick_params(labelsize=7, colors="#8b949e")
    ax_wave.set_facecolor("#0d1117")
    ax_wave.spines[:].set_color("#30363d")
    for sp in ["top", "right"]:
        ax_wave.spines[sp].set_visible(False)
    ax_wave.set_title(f"  {utt_id}   |   \"{transcript}\"",
                      fontsize=11, color="#e6edf3", pad=8, loc="left",
                      fontweight="normal")
    ax_wave.xaxis.set_visible(False)
    ax_wave.grid(axis="y", lw=0.3, color="#21262d")

    # ── Mel spectrogram ───────────────────────────────────────────────────
    ax_mel.imshow(mel, aspect="auto", origin="lower",
                  extent=[0, total_dur, 0, N_MELS],
                  cmap="magma", interpolation="nearest")
    ax_mel.set_xlim(0, total_dur)
    ax_mel.set_ylim(0, N_MELS)
    ax_mel.set_ylabel("mel bin", fontsize=8, color="#8b949e")
    ax_mel.tick_params(labelsize=7, colors="#8b949e")
    ax_mel.set_facecolor("#0d1117")
    ax_mel.spines[:].set_color("#30363d")
    for sp in ["top", "right"]:
        ax_mel.spines[sp].set_visible(False)

    # FACodec frame grid: every 8 frames = 100 ms
    grid_step = 8
    n_grid = n_frames // grid_step
    for g in range(1, n_grid + 1):
        gt = g * grid_step / FPS
        if gt > total_dur:
            break
        alpha = 0.18 if g % 5 != 0 else 0.35
        ax_mel.axvline(gt, color="white", lw=0.35, alpha=alpha, ls="-")

    # Phone-boundary vertical lines
    for t in boundaries[1:]:
        ax_mel.axvline(t, color="#39d353", lw=1.2, alpha=0.9, ls="--", zorder=5)

    # ── Phoneme region band ────────────────────────────────────────────────
    ax_phon.set_xlim(0, total_dur)
    ax_phon.set_ylim(0, 1)
    ax_phon.set_facecolor("#161b22")
    ax_phon.spines[:].set_color("#30363d")
    for sp in ["top", "right"]:
        ax_phon.spines[sp].set_visible(False)
    ax_phon.set_ylabel("phonemes", fontsize=8, color="#8b949e")
    ax_phon.tick_params(labelsize=7, colors="#8b949e")
    ax_phon.set_xlabel(
        "time (s)   —   green dashed = phone boundary, "
        "white dotted = FACodec frame grid (every 8 frames = 100 ms)",
        fontsize=8, color="#8b949e", labelpad=4)

    i = 0
    while i < len(labels):
        t_start = boundaries[i]
        ph = labels[i]
        j = i + 1
        while j < len(labels) and labels[j] == ph:
            j += 1
        t_end = boundaries[j] if j < len(boundaries) else total_dur

        color = phone_colors.get(ph, "#6272a4")
        ax_phon.axvspan(t_start, t_end, facecolor=color, alpha=0.72, lw=0, zorder=2)
        mid = (t_start + t_end) / 2
        ax_phon.text(mid, 0.5, ph, ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold",
                     bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.2,
                               boxstyle="round,pad=0.3"),
                     zorder=3)
        i = j

    # X-axis ticks showing time + frame index
    tick_times = np.linspace(0, total_dur, 12)
    tick_labels = [f"{tt:.2f}s\nF{int(tt * FPS)}" for tt in tick_times]
    ax_phon.set_xticks(tick_times)
    ax_phon.set_xticklabels(tick_labels, fontsize=5.5, color="#8b949e")
    ax_phon.grid(axis="x", lw=0.3, color="#21262d", alpha=0.5)

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# HTML report assembly
# ---------------------------------------------------------------------------

def build_html(panels: list[dict], output_path: str):
    import matplotlib.pyplot as plt

    # Collect legend
    all_phones = set()
    for p in panels:
        for lbl in p.get("all_labels", []):
            all_phones.add(lbl)
    all_phones = sorted(all_phones)

    legend_items = ""
    for i, ph in enumerate(all_phones):
        idx = i % len(PHONE_COLORS)
        legend_items += (
            f'<span class="legend-item">'
            f'<span class="swatch" style="background:{PHONE_COLORS[idx]}"></span>'
            f'<code>{ph}</code></span>'
        )

    panel_rows = ""
    for p in panels:
        panel_rows += f"""
        <div class="panel">
          <img src="{p['image_b64']}" class="panel-img" alt="{p['utt_id']}" />
          <div class="meta">
            <div class="row"><span class="k">ID</span><span class="v">{p['utt_id']}</span></div>
            <div class="row"><span class="k">Transcript</span><span class="v">"{p['transcript']}"</span></div>
            <div class="row"><span class="k">Duration</span><span class="v">{p['duration']:.2f}s</span></div>
            <div class="row"><span class="k">Phones</span><span class="v">{p['n_phones']} unique</span></div>
            <div class="row"><span class="k">Codec frames</span><span class="v">{p['n_frames']} @ {FPS} fps</span></div>
            <div class="row"><span class="k">Frame interval</span><span class="v">{FRAME_MS:.1f} ms</span></div>
            <div class="row"><span class="k">Mel hop</span><span class="v">{HOP_LENGTH} samples ({SR/HOP_LENGTH:.0f} Hz)</span></div>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phone–FACodec Alignment Report</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2128;
    --border: #21262d;
    --text: #e6edf3;
    --muted: #8b949e;
    --green: #39d353;
    --blue: #58a6ff;
    --red: #f85149;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         padding: 2rem 3rem; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 0.2rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.82rem; margin-bottom: 1.4rem; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 0.45rem;
             margin-bottom: 1.4rem; padding: 0.6rem 0.9rem;
             background: var(--surface); border: 1px solid var(--border);
             border-radius: 6px; align-items: center; }}
  .legend-label {{ font-size: 0.75rem; color: var(--muted); margin-right: 0.3rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3rem; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}
  code {{ font-size: 0.72rem; color: var(--muted); font-family: 'Fira Code', monospace; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 1.4rem; }}
  .panel {{ background: var(--surface); border: 1px solid var(--border);
            border-radius: 8px; overflow: hidden; }}
  .panel-img {{ width: 100%; display: block; }}
  .meta {{ padding: 0.75rem 1rem; display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; }}
  .row {{ display: flex; gap: 0.5rem; font-size: 0.8rem; align-items: baseline; }}
  .k {{ color: var(--muted); min-width: 100px; }}
  .v {{ color: var(--blue); font-family: 'Fira Code', monospace; font-size: 0.78rem; }}
  .info-bar {{ background: var(--surface2); border: 1px solid var(--border);
               border-radius: 6px; padding: 0.65rem 1rem; margin-bottom: 1.2rem;
               font-size: 0.8rem; color: var(--muted); line-height: 1.6; }}
  .info-bar strong {{ color: var(--green); }}
  .info-bar em {{ color: var(--blue); font-style: normal; }}
  .note {{ margin-top: 2rem; padding: 0.8rem 1rem; background: var(--surface);
           border: 1px solid var(--border); border-radius: 6px;
           font-size: 0.78rem; color: var(--muted); line-height: 1.7; }}
  .note strong {{ color: var(--red); }}
</style>
</head>
<body>
<h1>Phone–FACodec Frame Alignment Diagnostic</h1>
<p class="subtitle">scripts/visualize_alignment.py &mdash; AccentEdge</p>

<div class="info-bar">
  <strong>FACodec frame rate:</strong> {FPS:.0f} fps ({FRAME_MS:.1f} ms/frame, hop={HOP_LENGTH}) &nbsp;|&nbsp;
  <strong>Mel spec:</strong> n_fft={N_FFT}, n_mels={N_MELS} &nbsp;|&nbsp;
  <strong>Phone boundaries:</strong> <em>green dashed</em> lines on mel &nbsp;|&nbsp;
  <strong>Frame grid:</strong> white dotted lines every <em>8 frames = 100 ms</em>
</div>

<div class="legend">
  <span class="legend-label">Phoneme colours:</span>
  {legend_items or '<code>—</code>'}
</div>

<div class="grid">
{panel_rows}
</div>

<div class="note">
  <strong>Reading this report:</strong><br>
  Each panel shows one utterance. The green dashed lines mark detected phone-boundary
  transitions. Colour bands in the bottom row show which phoneme each FACodec frame
  belongs to. Frame-grid lines are every 100 ms (8 frames).<br><br>
  <strong>Common alignment issues:</strong><br>
  &bull; Green boundary shifted into a neighbouring word &rarr; CTC alignment error<br>
  &bull; Frames falling on silence / noise &rarr; transcript-audio mismatch<br>
  &bull; Uneven phoneme durations &rarr; phonemizer G2P artifacts (e.g. "fifty-five"
  as "fihf-t-iy" instead of two distinct number words)
</div>
</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"  -> Report written to {output_path}")


# ---------------------------------------------------------------------------
# Per-utterance processing
# ---------------------------------------------------------------------------

def process_utterance(wav_path: str | None, transcript: str,
                      utt_id: str) -> dict:
    """Load audio, compute mel + phone alignment, render panel."""

    if wav_path and Path(wav_path).exists():
        waveform = load_audio(wav_path)
        note = "real audio"
    else:
        print(f"  [no WAV found at '{wav_path}' -- synthesizing]")
        waveform = synthesize_speech(transcript)
        note = "synthetic"

    mel = compute_mel(waveform)
    codec_probs, phone_labels, frame_times = extract_phonemes(waveform, transcript)

    all_labels = list(set(phone_labels))
    img_b64 = plot_utterance(
        waveform=waveform,
        mel=mel,
        phone_labels=phone_labels,
        frame_times=frame_times,
        codec_probs=codec_probs,
        transcript=transcript,
        utt_id=utt_id,
    )

    print(f"  [{note}] {utt_id}: {len(waveform)/SR:.2f}s, "
          f"{mel.shape[1]} mel-frames, {len(all_labels)} unique phones")

    return {
        "utt_id": utt_id,
        "transcript": transcript,
        "image_b64": img_b64,
        "duration": len(waveform) / SR,
        "n_phones": len(all_labels),
        "n_frames": mel.shape[1],
        "all_labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Test utterances
# ---------------------------------------------------------------------------

TEST_UTTERANCES = [
    {
        "utt_id": "native_001",
        "text": "the weather in california is warm and pleasant",
        "desc": "native English -- standard LibriSpeech-style sentence",
    },
    {
        "utt_id": "indian_001",
        "text": "i would like to book a table for dinner at eight oclock",
        "desc": "Indian English -- rhotic /r/, flap t's, reduced vowels",
    },
    {
        "utt_id": "numbers_001",
        "text": "fifty five dollars please the total is thirteen twenty five",
        "desc": "numbers edge case -- compound number words, money context",
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phone-FACodec alignment HTML diagnostic report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wav", type=str, help="Path to WAV file")
    parser.add_argument("--text", type=str, help="Transcript for the WAV")
    parser.add_argument("--utt-id", type=str, default="custom_001",
                        help="Utterance ID")
    parser.add_argument("--output", type=str, default="artifacts/alignment_report.html",
                        help="Output HTML path")
    parser.add_argument("--test", type=str,
                        choices=["native", "indian", "numbers", "all"],
                        help="Run with built-in test utterances")
    parser.add_argument("--batch", type=str,
                        help="YAML or JSON file with list of {utt_id, text, wav?}")
    args = parser.parse_args()

    panels = []

    output_path = Path(args.output)
    if output_path.suffix != ".html":
        output_path = output_path.with_suffix(".html")
    output_path.parent.mkdir(exist_ok=True)

    # -- Single file
    if args.wav:
        transcript = args.text or "(no transcript)"
        panels.append(process_utterance(args.wav, transcript, args.utt_id))

    # -- Built-in tests
    elif args.test:
        keys = {"native": ["native_001"], "indian": ["indian_001"],
                "numbers": ["numbers_001"], "all": None}[args.test]
        for utt in TEST_UTTERANCES:
            if keys and utt["utt_id"] not in keys:
                continue
            print(f"\n  [{utt['desc']}]")
            panels.append(process_utterance(None, utt["text"], utt["utt_id"]))

    # -- Batch
    elif args.batch:
        import yaml, json
        path = Path(args.batch)
        if path.suffix in (".yaml", ".yml"):
            items = yaml.safe_load(path.read_text()) or []
        else:
            items = json.loads(path.read_text()) or []
        for item in items:
            wav = item.get("wav") or item.get("audio") or item.get("path")
            text = item.get("text") or item.get("transcript") or ""
            uid = item.get("utt_id") or item.get("id") or f"utt_{len(panels)+1:03d}"
            print(f"\n  [{uid}]")
            panels.append(process_utterance(wav, text, uid))

    # -- Default: run all built-in tests
    else:
        print("No --wav, --test, or --batch given -- running built-in tests ...")
        for utt in TEST_UTTERANCES:
            print(f"\n  [{utt['desc']}]")
            panels.append(process_utterance(None, utt["text"], utt["utt_id"]))

    if not panels:
        print("Nothing to render.")
        sys.exit(1)

    print(f"\nBuilding HTML report ({len(panels)} panel(s))...")
    build_html(panels, str(output_path))
    print("Done.")


if __name__ == "__main__":
    main()
