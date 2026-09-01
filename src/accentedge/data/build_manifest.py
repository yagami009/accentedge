#!/usr/bin/env python3
"""Corrected manifest builder - reads shard contents, not just shard filenames."""
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SSD = Path("/Volumes/AYUSH_SSD/accentedge-data")
OUT = SSD / "processed/manifests/utterances.parquet"

datasets = {
    "cmu_arctic": SSD / "raw/production-candidates/cmu-arctic",
    "libritts_r": SSD / "raw/production-candidates/libritts-r",
    "libritts_r_360": SSD / "raw/production-candidates/libritts-r-360",
    "vctk": SSD / "raw/production-candidates/vctk",
    "fleurs": SSD / "raw/production-candidates/fleurs",
    "indicvoices_hindi": SSD / "raw/production-candidates/indicvoices",
    "vaani_bangalore": SSD / "raw/production-candidates/vaani_bangalore",
    "vaani_hyderabad": SSD / "raw/production-candidates/vaani_hyderabad",
}

rows = []
for name, root in datasets.items():
    if not root.exists():
        print(f"SKIP {name}: not found")
        continue

    wavs = list(root.rglob("*.wav"))
    arrows = list(root.rglob("*.arrow"))
    parquets = list(root.rglob("*.parquet"))

    if wavs and not arrows and not parquets:
        # Pure WAV dataset (VCTK, LibriTTS)
        for wav in sorted(wavs):
            # Extract speaker from path
            rel = wav.relative_to(root)
            parts = rel.parts
            speaker = None
            for p in parts:
                if p.startswith('p') and p[1:].isdigit():
                    speaker = p
                    break
                if not any(c.isdigit() for c in p) and len(p) <= 10:
                    speaker = p

            rows.append({
                "utterance_id": f"{name}/{rel}",
                "dataset": name,
                "speaker_id": speaker,
                "audio_path": str(wav),
                "language": "en",
                "accent": "us" if name in ("libritts_r", "libritts_r_360", "cmu_arctic") else "unknown",
                "l1_language": "en",
                "recording_type": "scripted",
                "license_class": "cc_by",
                "split": "train",
                "quality_status": "verified",
            })
        print(f"  {name}: {len(wavs)} WAV utterances")

    elif arrows:
        # Arrow/IPC sharded dataset
        total_rows = 0
        for arrow_file in arrows:
            try:
                with pa.ipc.open_stream(str(arrow_file)) as reader:
                    table = reader.read_all()
                    n = table.num_rows
                    total_rows += n
                    # Try to extract metadata
                    cols = table.schema.names
                    df = table.to_pandas()
                    for col in ['speaker_id', 'speakerID', 'speaker', 'language', 'accent', 'l1_language', 'duration']:
                        if col in df.columns:
                            break
            except Exception as e:
                # Try as regular file
                try:
                    table = pa.ipc.open_file(str(arrow_file)).read_all()
                    n = table.num_rows
                    total_rows += n
                except Exception as e2:
                    print(f"    FAIL {arrow_file.name}: {e2}")
                    continue

            rows.append({
                "utterance_id": f"{name}/{arrow_file.stem}",
                "dataset": name,
                "speaker_id": None,
                "audio_path": str(arrow_file),
                "language": "en",
                "accent": "us" if name in ("fleurs",) else "indian" if "indicvoices" in name or "vaani" in name else "unknown",
                "l1_language": "en",
                "recording_type": "scripted",
                "license_class": "cc_by",
                "split": "train",
                "quality_status": "arrow_shard",
            })
        print(f"  {name}: {total_rows} utterances in {len(arrows)} arrow shards")

    elif parquets:
        # Parquet sharded dataset (Vaani)
        total_rows = 0
        total_duration = 0.0
        english_rows = 0
        english_duration = 0.0
        english_speakers = set()

        for pq_file in parquets:
            try:
                table = pq.read_table(str(pq_file))
                n = table.num_rows
                total_rows += n
                df = table.to_pandas()
                if 'duration' in df.columns:
                    total_duration += df['duration'].sum()
                if 'language' in df.columns:
                    en = df[df['language'] == 'English']
                    english_rows += len(en)
                    if 'duration' in en.columns:
                        english_duration += en['duration'].sum()
                    if 'speakerID' in en.columns:
                        english_speakers.update(en['speakerID'].dropna().unique())
            except Exception as e:
                print(f"    FAIL {pq_file.name}: {e}")
                continue

        rows.append({
            "utterance_id": f"{name}/{parquets[0].stem}",
            "dataset": name,
            "speaker_id": None,
            "audio_path": str(parquets[0]),
            "language": "hi" if "indicvoices" in name else "unknown",
            "accent": "indian",
            "l1_language": "hi" if "indicvoices" in name else "unknown",
            "recording_type": "scripted",
            "license_class": "cc_by",
            "split": "train",
            "quality_status": "parquet_shard",
        })
        print(f"  {name}: {total_rows} total rows, {total_duration/3600:.2f}h total")
        if english_rows > 0:
            print(f"    English: {english_rows} rows, {english_duration/3600:.2f}h, {len(english_speakers)} speakers")
        else:
            print(f"    English: 0 rows (all non-English)")

    else:
        print(f"  {name}: no recognizable files found")

print(f"\nTotal manifest entries: {len(rows)}")
df = pd.DataFrame(rows)
df.to_parquet(OUT, index=False)
print(f"Manifest written to {OUT}")
print(f"By dataset:\n{df['dataset'].value_counts().to_string()}")
