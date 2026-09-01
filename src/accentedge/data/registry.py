"""Data registry — maps dataset IDs to source locations and metadata."""

from pathlib import Path

SSD = Path("/Volumes/AYUSH_SSD/accentedge-data")

DATASETS = {
    "cmu_arctic": {
        "root": SSD / "raw/production-candidates/cmu-arctic",
        "type": "wav",
        "accent": "us",
        "language": "en",
        "license": "cmu_arctic",
    },
    "l2_arctic": {
        "root": SSD / "raw/research-only/l2-arctic",
        "type": "wav",
        "accent": "foreign",
        "language": "en",
        "license": "l2_arctic",
    },
    "libritts_r": {
        "root": SSD / "raw/production-candidates/libritts-r",
        "type": "wav",
        "accent": "us",
        "language": "en",
        "license": "libritts_r",
    },
    "vctk": {
        "root": SSD / "raw/production-candidates/vctk",
        "type": "wav",
        "accent": "mixed",
        "language": "en",
        "license": "vctk",
    },
    "fleurs": {
        "root": SSD / "raw/production-candidates/fleurs",
        "type": "hf",
        "accent": "indian",
        "language": "en",
        "license": "fleurs",
    },
    "indicvoices_hindi": {
        "root": SSD / "raw/production-candidates/indicvoices",
        "type": "parquet",
        "accent": "indian",
        "language": "hi",
        "license": "indicvoices_hindi",
    },
    "vaani": {
        "root": SSD / "raw/production-candidates/vaani",
        "type": "parquet",
        "accent": "indian",
        "language": "hi",
        "license": "vaani",
    },
}


def get_dataset(dataset_id: str) -> dict:
    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_id}")
    return DATASETS[dataset_id]
