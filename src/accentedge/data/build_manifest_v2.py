#!/usr/bin/env python3
"""Build two-level manifest: shards.parquet + utterances.parquet.
Uses vectorized pandas for speed. Handles Arrow streaming + Parquet.
"""
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import json

SSD = Path("/Volumes/AYUSH_SSD/accentedge-data")
OUT_DIR = SSD / "processed/manifests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "cmu_arctic": {"root": SSD / "raw/production-candidates/cmu-arctic", "role": "US_REFERENCE", "language": "en", "accent": "us", "region": "us", "col_map": {"speaker": "speaker_id", "file": "file_name", "text": "transcript"}},
    "libritts_r": {"root": SSD / "raw/production-candidates/libritts-r", "role": "NATIVE_ENGLISH_REFERENCE", "language": "en", "accent": "us", "region": "us", "wav": True},
    "libritts_r_360": {"root": SSD / "raw/production-candidates/libritts-r-360", "role": "NATIVE_ENGLISH_REFERENCE", "language": "en", "accent": "us", "region": "us", "wav": True},
    "vctk": {"root": SSD / "raw/production-candidates/vctk", "role": "MULTI_ACCENT_ENGLISH", "language": "en", "accent": "mixed", "region": "mixed", "wav": True},
    "fleurs": {"root": SSD / "raw/production-candidates/fleurs", "role": "NATIVE_ENGLISH_REFERENCE", "language": "en", "accent": "us", "region": "us"},
    "indicvoices_hindi": {"root": SSD / "raw/production-candidates/indicvoices", "role": "INDIC_ACOUSTIC_PRETRAINING", "language": "hi", "accent": "indian", "region": "in"},
    "vaani_bangalore": {"root": SSD / "raw/production-candidates/vaani_bangalore", "role": "INDIAN_ENGLISH_SOURCE", "language": "en", "accent": "indian", "region": "bangalore"},
    "vaani_hyderabad": {"root": SSD / "raw/production-candidates/vaani_hyderabad", "role": "INDIAN_ENGLISH_SOURCE", "language": "en", "accent": "indian", "region": "hyderabad"},
}

shard_rows = []
utt_batches = []
shard_counter = 0

for ds_id, cfg in DATASETS.items():
    root = cfg["root"]
    if not root.exists():
        print(f"SKIP {ds_id}: not found")
        continue
    wavs = sorted(root.rglob("*.wav"))
    arrows = sorted(root.rglob("*.arrow"))
    parquets = sorted(root.rglob("*.parquet"))
    col_map = cfg.get("col_map", {})
    is_wav = cfg.get("wav", False)

    if is_wav:
        # WAV-based datasets: one row per WAV file
        for wav in wavs:
            sid = f"{ds_id}__{shard_counter:06d}"
            shard_counter += 1
            rel = wav.relative_to(root)
            speaker = None
            for part in rel.parts:
                if part.startswith('p') and part[1:].isdigit():
                    speaker = part
                    break
                if not any(c.isdigit() for c in part) and len(part) <= 10:
                    speaker = part
            shard_rows.append({
                "shard_id": sid, "dataset_id": ds_id, "shard_path": str(wav),
                "format": "wav", "size_bytes": wav.stat().st_size,
                "num_utterances": 1, "duration_seconds": None,
                "language_counts": json.dumps({cfg["language"]: 1}),
                "speaker_count": 1, "scan_status": "ok",
            })
            utt_batches.append(pd.DataFrame([{
                "utterance_id": f"{ds_id}/{rel}",
                "dataset_id": ds_id, "shard_id": sid, "row_index": 0,
                "audio_locator": sid, "speaker_id": speaker,
                "language": cfg["language"], "accent": cfg["accent"],
                "region": cfg["region"], "l1_language": cfg["language"],
                "transcript": None, "duration_seconds": None,
                "sample_rate": 16000, "dataset_role": cfg["role"],
                "split": "train", "license_class": "cc_by",
                "quality_status": "wav",
            }]))
        print(f"  {ds_id}: {len(wavs)} WAV utterances")

    elif arrows:
        for arrow_path in arrows:
            sid = f"{ds_id}__{shard_counter:06d}"
            shard_counter += 1
            try:
                table = pa.ipc.open_stream(str(arrow_path)).read_all()
                n_rows = table.num_rows
                schema_names = table.schema.names
                # Duration sum - handle ChunkedArray
                dur_col = table.column("duration") if "duration" in schema_names else None
                total_dur = float(pc.sum(dur_col).as_py()) if dur_col is not None else None
                # Language counts
                lang_col = table.column("language") if "language" in schema_names else None
                langs = lang_col.to_pandas().value_counts().to_dict() if lang_col is not None else {cfg["language"]: n_rows}
                # Speaker count
                spk_col = table.column("speakerID") if "speakerID" in schema_names else (table.column("speaker") if "speaker" in schema_names else None)
                spk_count = int(spk_col.to_pandas().nunique()) if spk_col is not None else 0
            except Exception as e:
                n_rows = 0; total_dur = None; langs = {}; spk_count = 0
                print(f"    WARNING: shard metadata failed for {arrow_path.name}: {e}")
            shard_rows.append({
                "shard_id": sid, "dataset_id": ds_id, "shard_path": str(arrow_path),
                "format": "arrow", "size_bytes": arrow_path.stat().st_size,
                "num_utterances": n_rows, "duration_seconds": total_dur,
                "language_counts": json.dumps(langs), "speaker_count": spk_count,
                "scan_status": "ok" if n_rows > 0 else "error",
            })
            try:
                df = table.to_pandas()
                # Rename columns per dataset schema
                for old_col, new_col in col_map.items():
                    if old_col in df.columns:
                        df = df.rename(columns={old_col: new_col})
                df['_shard_id'] = sid
                df['_row_index'] = range(len(df))
                df['_ds_id'] = ds_id
                df['_accent'] = cfg["accent"]
                df['_region'] = cfg["region"]
                df['_def_lang'] = cfg["language"]
                if "language" not in df.columns:
                    df['language'] = cfg["language"]
                if "speaker_id" not in df.columns:
                    df['speaker_id'] = None
                if "duration" not in df.columns:
                    df['duration'] = None
                if "transcript" not in df.columns:
                    df['transcript'] = None
                batch = df[['_ds_id', '_shard_id', '_row_index', 'speaker_id', 'language',
                            '_accent', '_region', 'transcript', 'duration']].copy()
                if 'languagesKnown' in df.columns:
                    batch['l1_language'] = df['languagesKnown']
                else:
                    batch['l1_language'] = cfg.get('language', 'unknown')
                batch.columns = ['dataset_id', 'shard_id', 'row_index', 'speaker_id', 'language',
                                 'accent', 'region', 'l1_language', 'transcript', 'duration_seconds']
                batch['utterance_id'] = batch['dataset_id'] + '/' + arrow_path.stem + '__' + batch['row_index'].astype(str).str.zfill(6)
                batch['audio_locator'] = batch['shard_id']
                batch['sample_rate'] = 16000
                batch['dataset_role'] = cfg['role']
                batch['split'] = 'train'
                batch['license_class'] = 'cc_by'
                batch['quality_status'] = 'arrow_shard'
                utt_batches.append(batch)
            except Exception as e:
                print(f"    WARNING: failed utterances for {arrow_path.name}: {e}")
        en_count = sum(len(b[b['dataset_id'] == ds_id]) for b in utt_batches) if utt_batches else 0
        print(f"  {ds_id}: {len(arrows)} arrow shards, {en_count} English utterances")

    elif parquets:
        for pq_path in parquets:
            sid = f"{ds_id}__{shard_counter:06d}"
            shard_counter += 1
            try:
                df_shard = pd.read_parquet(pq_path)
                n_rows = len(df_shard)
                total_dur = float(df_shard['duration'].sum()) if 'duration' in df_shard.columns else None
                langs = df_shard['language'].value_counts().to_dict() if 'language' in df_shard.columns else {cfg["language"]: n_rows}
                spk_count = int(df_shard['speakerID'].nunique()) if 'speakerID' in df_shard.columns else 0
            except Exception as e:
                n_rows = 0; total_dur = None; langs = {}; spk_count = 0
                print(f"    WARNING: failed to read {pq_path.name}: {e}")
            shard_rows.append({
                "shard_id": sid, "dataset_id": ds_id, "shard_path": str(pq_path),
                "format": "parquet", "size_bytes": pq_path.stat().st_size,
                "num_utterances": n_rows, "duration_seconds": total_dur,
                "language_counts": json.dumps(langs), "speaker_count": spk_count,
                "scan_status": "ok" if n_rows > 0 else "error",
            })
            try:
                df = pd.read_parquet(pq_path)
                if 'language' in df.columns:
                    en_df = df[df['language'] == 'English'].copy()
                else:
                    en_df = df.copy()
                if len(en_df) == 0:
                    continue
                en_df['_shard_id'] = sid
                en_df['_row_index'] = range(len(en_df))
                en_df['_ds_id'] = ds_id
                if 'speakerID' not in en_df.columns:
                    en_df['speakerID'] = None
                if 'transcript' not in en_df.columns:
                    en_df['transcript'] = None
                if 'duration' not in en_df.columns:
                    en_df['duration'] = None
                batch = en_df[['_ds_id', '_shard_id', '_row_index', 'speakerID', 'language', 'duration', 'transcript']].copy()
                batch.columns = ['dataset_id', 'shard_id', 'row_index', 'speaker_id', 'language', 'duration_seconds', 'transcript']
                batch['utterance_id'] = batch['dataset_id'] + '/' + pq_path.stem + '__' + batch['row_index'].astype(str).str.zfill(6)
                batch['audio_locator'] = batch['shard_id']
                batch['accent'] = cfg['accent']
                batch['region'] = cfg['region']
                batch['l1_language'] = batch['language']
                batch['sample_rate'] = 16000
                batch['dataset_role'] = cfg['role']
                batch['split'] = 'train'
                batch['license_class'] = 'cc_by'
                batch['quality_status'] = 'parquet_shard'
                utt_batches.append(batch)
            except Exception as e:
                print(f"    WARNING: failed to enumerate {pq_path.name}: {e}")
        en_count = sum(len(b[b['dataset_id'] == ds_id]) for b in utt_batches) if utt_batches else 0
        print(f"  {ds_id}: {len(parquets)} parquet shards, {en_count} English utterances")

print("\nWriting shards.parquet...")
shards_df = pd.DataFrame(shard_rows)
shards_df.to_parquet(OUT_DIR / "shards.parquet", index=False)
print(f"Shards written: {len(shards_df)} rows")

print("Writing utterances per dataset...")
if utt_batches:
    for ds_id in sorted(DATASETS.keys()):
        ds_batches = [b for b in utt_batches if b['dataset_id'].iloc[0] == ds_id]
        if ds_batches:
            ds_df = pd.concat(ds_batches, ignore_index=True)
            ds_df['transcript'] = ds_df['transcript'].astype(str)
            ds_df['duration_seconds'] = pd.to_numeric(ds_df['duration_seconds'], errors='coerce')
            out_path = OUT_DIR / f"utterances_{ds_id}.parquet"
            ds_df.to_parquet(out_path, index=False)
            print(f"  {ds_id}: {len(ds_df)} rows")
else:
    print("  (no utterances)")

print("\n=== AUDIT SUMMARY ===")
print("Shards registry: shards.parquet")
print(f"Total shards: {len(shards_df)}")
for ds_id in sorted(shards_df["dataset_id"].unique()):
    ds_s = shards_df[shards_df["dataset_id"] == ds_id]
    print(f"\n{ds_id}: {len(ds_s)} shards")
    dur = ds_s["duration_seconds"].sum()
    if dur:
        print(f"  Duration: {dur/3600:.2f}h")
print("\nUtterance files: utterances_<dataset>.parquet")