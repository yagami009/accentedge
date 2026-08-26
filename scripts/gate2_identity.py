#!/usr/bin/env python3
"""
Gate 2 — Identity Preservation Calibration Check

Gate 2 Pass Criteria (identity preservation):
  • shift_over_span  < 0.25  (reconstruction damage is < 25% of natural speaker variation)
  • preservation_ratio  > 0.85  (reconstruction retains > 85% of same-speaker identity)

Uses the same three-reference calibration methodology as identity_comparison.py:
  1. SAME-SPEAKER reference  — within-speaker variation
  2. DIFFERENT-SPEAKER ref   — impostor floor
  3. RECONSTRUCTION ref      — source vs FACodec reconstruction

Key metric:
  shift_over_span = (same_speaker_median - recon_median) / same_speaker_span
  where span = same_speaker_median - impostor_median

Output: artifacts/gate2/identity_calibration.json
"""
from __future__ import annotations

import json
import os
import sys
import re
import warnings
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import librosa
import yaml
import types

warnings.filterwarnings("ignore")

# ═══════════════════════════════��═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

class Gate2Config:
    output_dir: Path = Path("artifacts/gate2")
    facodec_dir: Path = Path("/Users/ayushmh/FAcodec")
    facodec_ckpt: str = "Plachta/FAcodec"
    device: str = "cpu"
    n_pairs_ref: int = 40
    n_recon: int = 8
    seed: int = 42
    target_sr: int = 24000
    # Gate 2 thresholds
    max_shift_over_span: float = 0.25
    min_preservation: float = 0.85


cfg = Gate2Config()

# ═══════════════════════════════════════════════════════════════════════════════
# Mock audiotools
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock(name: str):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    return m


def install_mocks():
    mock_audio = _make_mock("audiotools")
    mock_ml = _make_mock("audiotools.ml")
    mock_ml.BaseModel = type("BaseModel", (), {"INTERN": [], "EXTERN": []})
    mock_audio.ml = mock_ml
    mock_audio.AudioSignal = type("AudioSignal", (), {})
    mock_audio.STFTParams = type("STFTParams", (), {})
    mock_core = _make_mock("audiotools.core")
    mock_core.util = _make_mock("audiotools.core.util")
    sys.modules["audiotools"] = mock_audio
    sys.modules["audiotools.ml"] = mock_ml
    sys.modules["audiotools.core"] = mock_core
    sys.modules["audiotools.core.util"] = mock_core.util


install_mocks()

# ═══════════════════════════════════════════════════════════════════════════════
# ECAPA-TDNN Speaker Embedding
# ═══════════════════════════════════════════════════════════════════════════════

class ECAPAEmbedding:
    """ECAPA-TDNN speaker embedding via SpeechBrain spkrec-ecapa-voxceleb."""

    def __init__(self, device: str = "cpu"):
        from speechbrain.pretrained import EncoderClassifier
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/spkrec_ecapa",
            run_opts={"device": device},
        )
        self.device = device

    @torch.no_grad()
    def __call__(self, wav: np.ndarray, sr: int = 24000) -> np.ndarray:
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0)
        emb = self.classifier.encode_batch(wav_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().cpu().numpy()

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


# ═══════════════════════════════════════════════════════════════════════════════
# FACodec Model
# ════════════════════════════════════��══════════════════════════════════════════

class FACodecModel:
    """Loaded FAcodec with encode / decode / reconstruct."""

    def __init__(self, facodec_dir: Path, ckpt_name: str, device: str = "cpu"):
        self.device = torch.device(device)

        if str(facodec_dir) not in sys.path:
            sys.path.insert(0, str(facodec_dir))
        modules_dir = facodec_dir / "modules"
        if modules_dir.exists() and str(modules_dir) not in sys.path:
            sys.path.insert(0, str(modules_dir))

        from modules.commons import build_model, recursive_munch
        from hf_utils import load_custom_model_from_hf

        ckpt_path, config_path = load_custom_model_from_hf(ckpt_name)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        mp = recursive_munch(config["model_params"])
        self.model = build_model(mp)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = ckpt.get("net", ckpt)
        for key in ckpt:
            self.model[key].load_state_dict(ckpt[key])
            self.model[key].eval().to(self.device)
            for p in self.model[key].parameters():
                p.requires_grad = False

    @torch.no_grad()
    def reconstruct(self, wav_np: np.ndarray) -> np.ndarray:
        """Encode -> decode round-trip."""
        wav_t = torch.from_numpy(wav_np).float()
        if wav_t.dim() == 1:
            wav_t = wav_t.unsqueeze(0)
        wav_in = wav_t.unsqueeze(0).to(self.device)
        z = self.model["encoder"](wav_in)
        z_q, _, _, _, _ = self.model["quantizer"](z, wav_in, n_c=2)
        recon = self.model["decoder"](z_q)
        return recon.squeeze().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Loading (reuses same functions as identity_comparison.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _resample(wav, orig_sr, target_sr):
    return librosa.resample(
        np.asarray(wav, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr
    ).astype(np.float32)


class Utt:
    __slots__ = ("speaker_id", "sentence_id", "text", "wav", "sr", "corpus")

    def __init__(self, speaker_id, sentence_id, text, wav, sr, corpus):
        self.speaker_id = speaker_id
        self.sentence_id = sentence_id
        self.text = text
        self.wav = wav
        self.sr = sr
        self.corpus = corpus


def load_cmu_arctic(cfg) -> list[Utt]:
    utterances: list[Utt] = []

    # Method 1: HuggingFace
    try:
        from datasets import load_dataset
        print("  Loading CMU ARCTIC from HuggingFace...")
        ds = load_dataset("cmu_arctic", split="train", trust_remote_code=True)

        speakers = sorted(set(ds["speaker_id"]))
        np.random.seed(cfg.seed)
        np.random.shuffle(speakers)
        chosen = speakers[:4]

        for spk in chosen:
            rows = [r for r in ds if r["speaker_id"] == spk]
            np.random.shuffle(rows)
            for row in rows[: cfg.n_recon + 4]:
                arr = row["audio"]["array"]
                sr = row["audio"]["sampling_rate"]
                wav = _resample(arr, sr, cfg.target_sr)
                utterances.append(Utt(
                    spk,
                    row.get("sentence_id", row.get("id", "")),
                    row.get("text", row.get("transcription", "")),
                    wav, cfg.target_sr, "native",
                ))
        print(f"  CMU ARCTIC: {len(utterances)} utterances, speakers {chosen}")
        return utterances
    except Exception as e:
        print(f"  [WARN] HF CMU ARCTIC failed: {e}")

    # Method 2: torchaudio
    try:
        import torchaudio
        print("  Loading CMU ARCTIC via torchaudio...")
        arctic_root = Path("/tmp/cmu_arctic")
        arctic_root.mkdir(exist_ok=True)

        speakers = ["slt", "bdl", "rms", "clb"]
        for spk in speakers:
            try:
                ds = torchaudio.datasets.CMU_ARCTIC(
                    root=str(arctic_root), speaker=spk, download=True,
                )
                idxs = np.random.choice(
                    len(ds), size=min(cfg.n_recon + 4, len(ds)), replace=False,
                )
                for idx in idxs:
                    wav, sr, transcript = ds[idx]
                    wav_np = wav.squeeze().numpy()
                    if sr != cfg.target_sr:
                        wav_np = _resample(wav_np, sr, cfg.target_sr)
                    utterances.append(Utt(
                        spk, str(idx), transcript, wav_np, cfg.target_sr, "native",
                    ))
            except Exception as e2:
                print(f"  [WARN] Speaker {spk} failed: {e2}")
        print(f"  CMU ARCTIC (torchaudio): {len(utterances)} utterances")
        return utterances
    except Exception as e:
        print(f"  [ERROR] All CMU ARCTIC loading methods failed: {e}")
        return utterances


def load_l2_arctic(cfg) -> list[Utt]:
    utterances: list[Utt] = []

    try:
        from datasets import load_dataset
        print("  Loading L2-ARCTIC from HuggingFace...")
        ds = load_dataset("osCa/L2-ARCTIC", split="train", trust_remote_code=True)

        all_speakers = sorted(set(ds["speaker_id"]))
        np.random.seed(cfg.seed)
        np.random.shuffle(all_speakers)

        indian = [s for s in all_speakers if s.startswith("HI")]
        if not indian:
            print(f"  [WARN] No HI-prefixed speakers. Available: {all_speakers[:10]}")
            indian = all_speakers[:4]
        chosen = indian[:4]
        print(f"  L2-ARCTIC Indian speakers: {chosen}")

        for spk in chosen:
            rows = [r for r in ds if r["speaker_id"] == spk]
            np.random.shuffle(rows)
            for row in rows[: cfg.n_recon + 4]:
                try:
                    arr = row["audio"]["array"]
                    sr = row["audio"]["sampling_rate"]
                    wav = _resample(arr, sr, cfg.target_sr)
                    utterances.append(Utt(
                        spk,
                        row.get("sentence_id", row.get("id", "")),
                        row.get("transcription", row.get("text", "")),
                        wav, cfg.target_sr, "indian",
                    ))
                except Exception as e2:
                    print(f"  [WARN] Skip {spk} row: {e2}")
        print(f"  L2-ARCTIC: {len(utterances)} utterances")
        return utterances
    except Exception as e:
        print(f"  [ERROR] L2-ARCTIC loading failed: {e}")
        return utterances


# ═══════════════════════════════════════════════════════════════════════════════
# Reference distributions
# ═══════════════════════════════════════════════════════════════════════════════

def _summary_stats(values):
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_speaker_groups(utterances):
    groups = {}
    for u in utterances:
        groups.setdefault(u.speaker_id, []).append(u)
    return groups


# ── Prompt matching ────────────────────────────────────────────────────────────

def _normalize_text(text):
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_matched_prompts(native, indian):
    """Find utterances with matching transcripts across corpora."""
    native_by_text = {}
    for u in native:
        key = _normalize_text(u.text)
        if key and key not in native_by_text:
            native_by_text[key] = u

    matched_native, matched_indian = [], []
    seen = set()
    for u in indian:
        key = _normalize_text(u.text)
        if key in native_by_text:
            nu = native_by_text[key]
            if id(nu) not in seen:
                matched_native.append(nu)
                matched_indian.append(u)
                seen.add(id(nu))
    n = len(matched_native)
    print(f"  Matched prompts: {n} pairs")
    return matched_native, matched_indian


# ── Legacy SECS diagnostic ────────────────────────────────────────────────────

def compute_legacy_secs(utts, wavs, extractor):
    """Legacy mean source-reconstruction cosine sim. DIAGNOSTIC_ONLY."""
    sims = []
    for utt, recon in zip(utts, wavs):
        sims.append(extractor.cosine_sim(
            extractor(utt.wav, utt.sr),
            extractor(recon, utt.sr),
        ))
    return float(np.mean(sims)) if sims else float("nan")


def compute_same_speaker(utterances, extractor, n_pairs=40):
    """Within-speaker cosine sims."""
    groups = build_speaker_groups(utterances)
    sims = []
    for spk, utts in groups.items():
        if len(utts) < 2:
            continue
        for i in range(len(utts)):
            for j in range(i + 1, len(utts)):
                sims.append(extractor.cosine_sim(
                    extractor(utts[i].wav, utts[i].sr),
                    extractor(utts[j].wav, utts[j].sr),
                ))
                if len(sims) >= n_pairs:
                    break
            if len(sims) >= n_pairs:
                break
    return {"sims": sims, "summary": _summary_stats(sims)}


def compute_impostor(group_a, group_b, extractor, n_pairs=40):
    """Cross-speaker cosine sims (different speakers only)."""
    emb_a = [(u, extractor(u.wav, u.sr)) for u in group_a]
    emb_b = [(u, extractor(u.wav, u.sr)) for u in group_b]

    sims = []
    rng = np.random.RandomState(cfg.seed)
    for u_a, e_a in emb_a:
        candidates = [(u_b, e_b) for u_b, e_b in emb_b if u_b.speaker_id != u_a.speaker_id]
        for u_b, e_b in rng.permutation(candidates):
            sims.append(extractor.cosine_sim(e_a, e_b))
            if len(sims) >= n_pairs:
                break
        if len(sims) >= n_pairs:
            break
    return {"sims": sims, "summary": _summary_stats(sims)}


def compute_reconstruction(utts, wavs, extractor):
    """Source vs reconstruction cosine sims."""
    sims = []
    for utt, recon in zip(utts, wavs):
        try:
            sims.append(extractor.cosine_sim(
                extractor(utt.wav, utt.sr),
                extractor(recon, utt.sr),
            ))
        except Exception as e:
            print(f"  [WARN] Embedding failed for {utt.sentence_id}: {e}")
    return {"sims": sims, "summary": _summary_stats(sims)}


# ═══════════════��═══════════════════════════════════════════════════════════════
# Calibration & Gate Check
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate(same_spk, impostor, recon):
    """Compute calibrated metrics from three references."""
    ss_med = same_spk["summary"]["median"]
    imp_med = impostor["summary"]["median"]
    recon_med = recon["summary"]["median"]

    span = ss_med - imp_med
    shift = ss_med - recon_med
    imp_dist = recon_med - imp_med

    shift_over_span = shift / span if span > 1e-9 else float("nan")
    preservation = recon_med / ss_med if ss_med > 1e-9 else float("nan")

    return {
        "same_speaker_median": float(ss_med),
        "impostor_median": float(imp_med),
        "recon_median": float(recon_med),
        "same_speaker_span": float(span),
        "reconstruction_shift": float(shift),
        "shift_over_span": float(shift_over_span),
        "preservation_ratio": float(preservation),
    }


def check_gate2(cal_native, cal_indian, cfg):
    """Gate 2 pass criteria."""
    results = {}

    for corpus, cal in [("native", cal_native), ("indian", cal_indian)]:
        sos = cal["shift_over_span"]
        pres = cal["preservation_ratio"]

        shift_ok = not np.isnan(sos) and sos < cfg.max_shift_over_span
        pres_ok = not np.isnan(pres) and pres > cfg.min_preservation
        passed = shift_ok and pres_ok

        results[corpus] = {
            "shift_over_span": float(sos) if not np.isnan(sos) else None,
            "preservation_ratio": float(pres) if not np.isnan(pres) else None,
            "shift_over_span_pass": bool(shift_ok),
            "preservation_ratio_pass": bool(pres_ok),
            "gate2_pass": bool(passed),
            "thresholds": {
                "max_shift_over_span": cfg.max_shift_over_span,
                "min_preservation": cfg.min_preservation,
            },
        }

    overall = results["native"]["gate2_pass"] and results["indian"]["gate2_pass"]
    results["overall"] = {
        "gate2_pass": bool(overall),
        "interpretation": _interpret(results),
    }
    return results


def _interpret(results):
    parts = []
    for corpus in ["native", "indian"]:
        r = results[corpus]
        status = "PASS" if r["gate2_pass"] else "FAIL"
        sos = f"shift/span={r['shift_over_span']:.4f}" if r["shift_over_span"] is not None else "shift/span=N/A"
        pres = f"preservation={r['preservation_ratio']:.4f}" if r["preservation_ratio"] is not None else "preservation=N/A"
        parts.append(f"{corpus}: {status} ({sos}, {pres})")
    return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import re as _re

    t_start = time.time()
    np.random.seed(cfg.seed)

    parser = argparse.ArgumentParser(description="Gate 2 — Identity Preservation Calibration")
    parser.add_argument("--facodec-dir", type=str, default=str(cfg.facodec_dir))
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--n-pairs", type=int, default=cfg.n_pairs_ref)
    parser.add_argument("--n-recon", type=int, default=cfg.n_recon)
    parser.add_argument("--output-dir", type=str, default=str(cfg.output_dir))
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    cfg.facodec_dir = Path(args.facodec_dir)
    cfg.device = args.device
    cfg.n_pairs_ref = args.n_pairs
    cfg.n_recon = args.n_recon
    cfg.output_dir = Path(args.output_dir)
    cfg.seed = args.seed

    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output:  {cfg.output_dir}")
    print(f"Device:  {cfg.device}")
    print(f"Seed:    {cfg.seed}")

    # Load models
    print("\n>>> Loading ECAPA-TDNN...")
    extractor = ECAPAEmbedding(device=cfg.device)
    print(f"  ECAPA-TDNN ready on {cfg.device}")

    if not cfg.facodec_dir.exists():
        print(f"\n[ERROR] FACodec directory not found: {cfg.facodec_dir}")
        sys.exit(1)

    print(f"\n>>> Loading FACodec ({cfg.facodec_ckpt})...")
    facodec = FACodecModel(cfg.facodec_dir, cfg.facodec_ckpt, device=cfg.device)
    print("  FACodec ready")

    # Load datasets
    print("\n>>> Loading datasets...")
    native_utts = load_cmu_arctic(cfg)
    indian_utts = load_l2_arctic(cfg)

    if len(native_utts) < 4:
        print("[ERROR] Fewer than 4 native utterances — aborting.")
        sys.exit(1)
    if len(indian_utts) < 4:
        print("[ERROR] Fewer than 4 Indian utterances — aborting.")
        sys.exit(1)

    # Three-reference calibration
    print("\n>>> Building three-reference distributions...")

    # 1. Same-speaker
    native_ss = compute_same_speaker(native_utts, extractor, cfg.n_pairs_ref)
    indian_ss = compute_same_speaker(indian_utts, extractor, cfg.n_pairs_ref)
    print(f"  Native same-speaker:   n={native_ss['summary']['n']}  "
          f"median={native_ss['summary']['median']:.4f}")
    print(f"  Indian same-speaker:   n={indian_ss['summary']['n']}  "
          f"median={indian_ss['summary']['median']:.4f}")

    # 2. Impostor
    native_imp = compute_impostor(native_utts, native_utts, extractor, cfg.n_pairs_ref)
    indian_imp = compute_impostor(indian_utts, indian_utts, extractor, cfg.n_pairs_ref)
    print(f"  Native impostor:       n={native_imp['summary']['n']}  "
          f"median={native_imp['summary']['median']:.4f}")
    print(f"  Indian impostor:       n={indian_imp['summary']['n']}  "
          f"median={indian_imp['summary']['median']:.4f}")

    # 3. Reconstruction
    native_recon_wavs, indian_recon_wavs = [], []
    for utt in native_utts[:cfg.n_recon]:
        try:
            recon = facodec.reconstruct(utt.wav)
            native_recon_wavs.append(recon)
        except Exception as e:
            print(f"  [WARN] Native recon failed {utt.sentence_id}: {e}")
    for utt in indian_utts[:cfg.n_recon]:
        try:
            recon = facodec.reconstruct(utt.wav)
            indian_recon_wavs.append(recon)
        except Exception as e:
            print(f"  [WARN] Indian recon failed {utt.sentence_id}: {e}")

    native_recon = compute_reconstruction(
        native_utts[:len(native_recon_wavs)], native_recon_wavs, extractor,
    )
    indian_recon = compute_reconstruction(
        indian_utts[:len(indian_recon_wavs)], indian_recon_wavs, extractor,
    )
    print(f"  Native recon:          n={native_recon['summary']['n']}  "
          f"median={native_recon['summary']['median']:.4f}")
    print(f"  Indian recon:          n={indian_recon['summary']['n']}  "
          f"median={indian_recon['summary']['median']:.4f}")

    # Legacy SECS diagnostic (preserved, labeled DIAGNOSTIC_ONLY)
    native_legacy = compute_legacy_secs(
        native_utts[:len(native_recon_wavs)], native_recon_wavs, extractor,
    )
    indian_legacy = compute_legacy_secs(
        indian_utts[:len(indian_recon_wavs)], indian_recon_wavs, extractor,
    )
    print(f"\n  +{'-'*56}+")
    print(f"  |  Legacy SECS mean (DIAGNOSTIC ONLY)              |")
    print(f"  |  Native: {native_legacy:.4f}                                  |")
    print(f"  |  Indian: {indian_legacy:.4f}                                  |")
    print(f"  |  Broken-adapter range: 0.05 - 0.24               |")
    print(f"  |  NOT valid FACodec performance                   |")
    print(f"  +{'-'*56}+")

    # Calibration
    cal_native = calibrate(native_ss, native_imp, native_recon)
    cal_indian = calibrate(indian_ss, indian_imp, indian_recon)

    # Gate 2 check
    gate = check_gate2(cal_native, cal_indian, cfg)

    # Print results
    print("\n" + "=" * 64)
    print("  GATE 2 — IDENTITY PRESERVATION CALIBRATION")
    print("=" * 64)
    header = f"  {'Metric':<30} {'Native':>12} {'Indian':>12} {'Threshold':>10}"
    print(header)
    print("  " + "-" * 64)

    def show(name, nv, iv, thresh="", ok_n="", ok_i=""):
        def fmt(v):
            if v is None:
                return "N/A"
            return f"{v:.4f}"
        right = ""
        if ok_n or ok_i:
            right = f"  {ok_n} / {ok_i}"
        print(f"  {name:<30} {fmt(nv):>12} {fmt(iv):>12} {thresh:>10}{right}")

    show("same-speaker median",  cal_native["same_speaker_median"],
         cal_indian["same_speaker_median"])
    show("impostor median",       cal_native["impostor_median"],
         cal_indian["impostor_median"])
    show("recon median",          cal_native["recon_median"],
         cal_indian["recon_median"])
    show("same-speaker span",     cal_native["same_speaker_span"],
         cal_indian["same_speaker_span"])
    show("reconstruction shift",  cal_native["reconstruction_shift"],
         cal_indian["reconstruction_shift"])
    show("shift / span",          cal_native["shift_over_span"],
         cal_indian["shift_over_span"],
         f"< {cfg.max_shift_over_span}",
         "PASS" if gate["native"]["shift_over_span_pass"] else "FAIL",
         "PASS" if gate["indian"]["shift_over_span_pass"] else "FAIL")
    show("preservation ratio",    cal_native["preservation_ratio"],
         cal_indian["preservation_ratio"],
         f"> {cfg.min_preservation}",
         "PASS" if gate["native"]["preservation_ratio_pass"] else "FAIL",
         "PASS" if gate["indian"]["preservation_ratio_pass"] else "FAIL")
    print("  " + "-" * 64)
    print(f"  {'GATE 2 PASS':<30} {'YES' if gate['native']['gate2_pass'] else 'NO':>12} "
          f"{'YES' if gate['indian']['gate2_pass'] else 'NO':>12}")
    print(f"  {'OVERALL':<30} {'':>12} {'':>12} "
          f"{'PASS ✓' if gate['overall']['gate2_pass'] else 'FAIL ✗':>10}")
    print("=" * 64)
    print(f"\n  {gate['overall']['interpretation']}")
    print(f"  Elapsed: {time.time() - t_start:.1f}s")

    # Save output
    def _safe(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v

    output = {
        "gate": 2,
        "gate_name": "Identity Preservation Calibration",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "device": cfg.device,
            "seed": cfg.seed,
            "facodec_ckpt": cfg.facodec_ckpt,
            "n_pairs_ref": cfg.n_pairs_ref,
            "n_recon": cfg.n_recon,
            "max_shift_over_span": cfg.max_shift_over_span,
            "min_preservation": cfg.min_preservation,
        },
        "methodology": {
            "same_speaker_ref": "Within-speaker pairs: utterance A vs utterance B",
            "different_speaker_ref": "Cross-speaker pairs: speaker A vs speaker B",
            "reconstruction_ref": "Source waveform vs FACodec reconstruction",
            "key_metric": (
                "shift_over_span = reconstruction_shift / same_speaker_span. "
                "Reconstruction damage as fraction of natural speaker variation."
            ),
            "pass_criteria": (
                f"shift_over_span < {cfg.max_shift_over_span} "
                f"AND preservation_ratio > {cfg.min_preservation}"
            ),
        },
        "matched_prompts": matched_n,
        "native": {
            "calibration": {k: _safe(v) for k, v in cal_native.items()},
            "reference_distributions": {
                "same_speaker": {
                    "n": native_ss["summary"]["n"],
                    "median": _safe(native_ss["summary"]["median"]),
                    "mean": _safe(native_ss["summary"]["mean"]),
                },
                "impostor": {
                    "n": native_imp["summary"]["n"],
                    "median": _safe(native_imp["summary"]["median"]),
                    "mean": _safe(native_imp["summary"]["mean"]),
                },
                "reconstruction": {
                    "n": native_recon["summary"]["n"],
                    "median": _safe(native_recon["summary"]["median"]),
                    "mean": _safe(native_recon["summary"]["mean"]),
                },
            },
            "gate_result": gate["native"],
            "legacy_secs_diagnostic": {
                "native_mean": _safe(native_legacy),
                "indian_mean": _safe(indian_legacy),
                "label": "DIAGNOSTIC_ONLY / BROKEN ADAPTER / NOT VALID FACODEC PERFORMANCE",
                "known_broken_range": "0.05-0.24",
                "note": (
                    "This metric used arbitrary cosine-similarity thresholds. "
                    "It does NOT calibrate against speaker variation and is "
                    "unreliable for cross-corpus comparison. Superseded by "
                    "shift_over_span and preservation_ratio."
                ),
            },
        },
        "indian": {
            "calibration": {k: _safe(v) for k, v in cal_indian.items()},
            "reference_distributions": {
                "same_speaker": {
                    "n": indian_ss["summary"]["n"],
                    "median": _safe(indian_ss["summary"]["median"]),
                    "mean": _safe(indian_ss["summary"]["mean"]),
                },
                "impostor": {
                    "n": indian_imp["summary"]["n"],
                    "median": _safe(indian_imp["summary"]["median"]),
                    "mean": _safe(indian_imp["summary"]["mean"]),
                },
                "reconstruction": {
                    "n": indian_recon["summary"]["n"],
                    "median": _safe(indian_recon["summary"]["median"]),
                    "mean": _safe(indian_recon["summary"]["mean"]),
                },
            },
            "gate_result": gate["indian"],
        },
        "overall": gate["overall"],
    }

    out_path = cfg.output_dir / "identity_calibration.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    if gate["overall"]["gate2_pass"]:
        print("\n  ▸ GATE 2 PASSED — Identity preservation within acceptable bounds.")
    else:
        print("\n  ✗ GATE 2 FAILED — Identity preservation does not meet criteria.")
        print(f"    {gate['overall']['interpretation']}")

    return output


if __name__ == "__main__":
    main()
