#!/usr/bin/env python3
"""Automated teacher benchmark scoring — no human bias."""
import json, re
from pathlib import Path
import soundfile as sf
import numpy as np

SOURCE_DIR = Path("/Users/ayushmh/accentedge/experiment_zero/e0a_vaani/source_candidates")
EMILIA_DIR = Path("/Users/ayushmh/accentedge/experiment_zero/e0a_vaani/teacher_benchmark/cosyaccent")
PAPER_DIR = Path("/Users/ayushmh/accentedge/experiment_zero/e0a_vaani/teacher_benchmark/cosyaccent_paper")
TRANSCRIPT_FILE = SOURCE_DIR / "selected_sources.json"

with open(TRANSCRIPT_FILE) as f:
    meta = json.load(f)
sources = {s["clip_id"]: s for s in meta["sources"]}

def vad_duration(data, sr, threshold_db=-40):
    frame_size = int(0.025 * sr)
    hop_size = int(0.010 * sr)
    frames = []
    for i in range(0, len(data) - frame_size, hop_size):
        frame = data[i:i+frame_size]
        rms = np.sqrt(np.mean(frame**2))
        frames.append(rms)
    frames = np.array(frames)
    frames_db = 20 * np.log10(frames + 1e-10)
    active = np.sum(frames_db > threshold_db) * hop_size / sr
    return active

def clipping_metrics(data, threshold=0.99, min_run=3):
    clipped = np.sum(np.abs(data) >= threshold)
    flat_tops = 0
    run = 0
    for val in np.abs(data) >= threshold:
        if val:
            run += 1
            if run == min_run:
                flat_tops += 1
        else:
            run = 0
    return int(clipped), flat_tops

def estimate_f0(data, sr):
    frame_size = int(0.05 * sr)
    f0s = []
    for i in range(0, len(data) - frame_size, frame_size):
        frame = data[i:i+frame_size]
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        peak = np.argmax(corr[10:100]) + 10
        f0 = sr / peak if peak > 0 else 0
        if 50 < f0 < 500:
            f0s.append(f0)
    return np.mean(f0s) if f0s else 0

def spectral_centroid_shift(src_data, tgt_data, sr):
    def centroid(data, sr):
        frame_size = int(0.025 * sr)
        centroids = []
        for i in range(0, len(data) - frame_size, hop_size):
            frame = data[i:i+frame_size]
            mag = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1/sr)
            if mag.sum() > 0:
                centroids.append(np.sum(freqs * mag) / mag.sum())
        return np.mean(centroids) if centroids else 0
    hop_size = int(0.010 * sr)
    src_c = centroid(src_data, sr)
    tgt_c = centroid(tgt_data, sr)
    if src_c > 0:
        return (tgt_c - src_c) / src_c * 100
    return 0

def cer(ref, hyp):
    ref, hyp = ref.lower(), hyp.lower()
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    if len(ref) == 0:
        return 0.0
    return d[len(ref)][len(hyp)] / len(ref)

def auto_transcribe(wav_path):
    """Lightweight ASR using whisper CLI if available."""
    try:
        import whisper
        model = whisper.load_model("tiny.en")
        result = model.transcribe(str(wav_path))
        return result["text"].strip()
    except Exception:
        return ""

# Main scoring
results = []

for i in range(1, 6):
    src = SOURCE_DIR / f'vaani_src_00{i}.wav'
    emilia = EMILIA_DIR / f'cosyaccent_00{i}.wav'
    paper = PAPER_DIR / f'cosyaccent_paper_00{i}.wav'
    src_data, src_sr = sf.read(src)
    ref_text = sources[i]["transcript"]
    clean_ref = re.sub(r'<[^>]+>', '', ref_text).strip()

    for teacher, tgt_path in [("emilia", emilia), ("paper", paper)]:
        if not tgt_path.exists():
            continue
        tgt_data, tgt_sr = sf.read(tgt_path)

        raw_dur = len(tgt_data) / tgt_sr
        src_dur = len(src_data) / src_sr
        vad_dur = vad_duration(tgt_data, tgt_sr)
        src_vad = vad_duration(src_data, src_sr)
        vad_ratio = vad_dur / src_vad if src_vad > 0 else 0

        src_f0 = estimate_f0(src_data, src_sr)
        tgt_f0 = estimate_f0(tgt_data, tgt_sr)
        f0_shift = 12 * np.log2(tgt_f0 / src_f0) if src_f0 > 0 else 0

        centroid_shift = spectral_centroid_shift(src_data, tgt_data, src_sr)

        clip_count, flat_tops = clipping_metrics(tgt_data)

        hyp_text = auto_transcribe(tgt_path)
        cer_score = cer(clean_ref, hyp_text) if hyp_text else None

        # Derived scores
        duration_score = max(0, 10 - abs(vad_ratio - 1.0) * 20) if vad_ratio > 0.7 else 0
        if 0.90 <= vad_ratio <= 1.10:
            duration_score = 10
        elif 0.80 <= vad_ratio < 0.90 or 1.10 < vad_ratio <= 1.20:
            duration_score = 7
        elif 0.70 <= vad_ratio < 0.80 or 1.20 < vad_ratio <= 1.30:
            duration_score = 4
        else:
            duration_score = 1

        clipping_score = 10 if clip_count == 0 else (7 if clip_count < 20 else 3)

        content_score = 10 if cer_score and cer_score < 0.15 else (5 if cer_score and cer_score < 0.30 else 1)

        # Speaker similarity via simple MFCC + cosine (librosa if available)
        speaker_sim = None
        try:
            import librosa
            src_mfcc = librosa.feature.mfcc(y=src_data, sr=src_sr, n_mfcc=13).mean(axis=1)
            tgt_mfcc = librosa.feature.mfcc(y=tgt_data, sr=tgt_sr, n_mfcc=13).mean(axis=1)
            norm = np.linalg.norm(src_mfcc) * np.linalg.norm(tgt_mfcc)
            speaker_sim = float(np.dot(src_mfcc, tgt_mfcc) / norm) if norm > 0 else 0
        except ImportError:
            pass

        overall = (duration_score * 0.2 + clipping_score * 0.15 +
                   content_score * 0.25 + (speaker_sim or 0.5) * 10 * 0.2 + 5 * 0.2)

        results.append({
            "clip_id": i,
            "teacher": teacher,
            "source_file": src.name,
            "output_file": tgt_path.name,
            "reference_transcript": clean_ref,
            "asr_transcript": hyp_text,
            "raw_duration": round(raw_dur, 3),
            "vad_duration": round(vad_dur, 3),
            "vad_ratio": round(vad_ratio, 3),
            "duration_score": duration_score,
            "f0_shift_semitones": round(f0_shift, 2),
            "spectral_centroid_shift_pct": round(centroid_shift, 1),
            "speaker_similarity": round(speaker_sim, 4) if speaker_sim else None,
            "clipped_samples": clip_count,
            "flat_tops": flat_tops,
            "clipping_score": clipping_score,
            "cer": round(cer_score, 4) if cer_score else None,
            "content_score": content_score,
            "overall_score": round(overall, 2),
            "accent_movement_auto": round(centroid_shift / 10, 1),
            "decision": "APPROVED" if overall >= 6 and content_score >= 5 else "REJECTED"
        })

# Print summary
print("=" * 100)
print("AUTOMATED TEACHER BENCHMARK — NO HUMAN BIAS")
print("=" * 100)
print(f"{'Clip':<6} {'Teacher':<10} {'VAD ratio':<12} {'F0 shift':<12} {'Centroid':<12} {'Spk sim':<10} {'CER':<8} {'Clip':<8} {'Overall':<10} {'Decision'}")
print("-" * 100)
for r in results:
    clip_flag = "⚠️" if r['flat_tops'] > 5 else "✅"
    cer_str = f"{r['cer']:.3f}" if r['cer'] else "N/A"
    spk_str = f"{r['speaker_similarity']:.3f}" if r['speaker_similarity'] else "N/A"
    print(f"{r['clip_id']:<6} {r['teacher']:<10} {r['vad_ratio']:<12.3f} {r['f0_shift_semitones']:<+12.2f} {r['spectral_centroid_shift_pct']:<+12.1f} {spk_str:<10} {cer_str:<8} {clip_flag:<8} {r['overall_score']:<10.2f} {r['decision']}")

# Per-clip comparison
print("\n" + "=" * 100)
print("PER-CLIP WINNER")
print("=" * 100)
for i in range(1, 6):
    em = next((r for r in results if r['clip_id'] == i and r['teacher'] == 'emilia'), None)
    pa = next((r for r in results if r['clip_id'] == i and r['teacher'] == 'paper'), None)
    if em and pa:
        winner = "PAPER" if pa['overall_score'] > em['overall_score'] else "EMILIA"
        margin = abs(pa['overall_score'] - em['overall_score'])
        print(f"Clip {i}: {winner} wins (score: {pa['overall_score']:.2f} vs {em['overall_score']:.2f}, margin: {margin:.2f})")
        if pa['vad_ratio'] > 1.1:
            print(f"  ⚠️ Paper VAD ratio {pa['vad_ratio']:.2f} — slower than source")
        if em['vad_ratio'] < 0.8:
            print(f"  ⚠️ Emilia VAD ratio {em['vad_ratio']:.2f} — compresses speech")

# Aggregate
print("\n" + "=" * 100)
print("AGGREGATE")
print("=" * 100)
for teacher in ['emilia', 'paper']:
    t_results = [r for r in results if r['teacher'] == teacher]
    avg_overall = np.mean([r['overall_score'] for r in t_results])
    avg_vad = np.mean([r['vad_ratio'] for r in t_results])
    avg_cer = np.mean([r['cer'] for r in t_results if r['cer'] is not None])
    wins = sum(1 for i in range(1, 6) if
               next((r for r in results if r['clip_id'] == i and r['teacher'] == teacher), {}).get('overall_score', 0) >
               next((r for r in results if r['clip_id'] == i and r['teacher'] != teacher), {}).get('overall_score', 0))
    approved = sum(1 for r in t_results if r['decision'] == 'APPROVED')
    print(f"{teacher.upper()}: avg overall={avg_overall:.2f}, avg VAD={avg_vad:.3f}, avg CER={avg_cer:.3f}, approved={approved}/5")

# Save
out_path = Path("/Users/ayushmh/accentedge/experiment_zero/e0a_vaani/teacher_benchmark/auto_score_report.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nFull report: {out_path}")
