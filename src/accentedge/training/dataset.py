"""Phase 1 — Native-prior PyTorch Dataset.

Yields tuples of
    zc1, phone_ids, valid_frame_mask, speaker_id, item_id

where

    zc1              : [1, T]  int64  FACodec content codebook indices
    phone_ids        : [T]     int64  frame-level phoneme IDs at codec_fps
    valid_frame_mask : [T]     bool   True for real frames, False for padding
    speaker_id       : str            speaker identifier
    item_id          : str            unique utterance identifier

Caching: extracted latents and phone IDs are cached to *cache_dir* so the
frozen FACodec model and phoneme pipeline are only run once per file.

Config keys read (all under ``dataset`` block):

    dataset:
        sample_rate : int   (default 24000)
        codec_hop   : int   (default 300)
        codec_fps   : int   (default 80)
        facodec_device : str (default "cpu")
        normalize_latents : bool (default False)

Additionally, top-level keys are consulted:

    paths:
        facodec_checkpoint : str
        cache_dir          : str
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from accentedge.codec.facodec import FACodecAdapter
from accentedge.phase1.phoneme_pipeline import PhonemePipeline


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, item_id: str) -> Path:
    """Return the directory where a single item's cached data is stored."""
    h = hashlib.sha256(item_id.encode()).hexdigest()[:16]
    return cache_dir / h[:2] / h


def _load_cached(cache_dir: Path, item_id: str) -> Optional[Dict[str, Any]]:
    """Load cached latents / phone IDs for an item, or None if not cached."""
    d = _cache_path(cache_dir, item_id)
    latents_file = d / "latents.pt"
    phones_file = d / "phone_ids.pt"
    meta_file = d / "metadata.json"

    if not (latents_file.exists() and phones_file.exists() and meta_file.exists()):
        return None

    try:
        zc1 = torch.load(latents_file, map_location="cpu", weights_only=True)
        phone_ids = torch.load(phones_file, map_location="cpu", weights_only=True)
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        return {
            "zc1": zc1,
            "phone_ids": phone_ids,
            "speaker_id": metadata.get("speaker_id", ""),
            "item_id": metadata.get("item_id", item_id),
            "num_frames": metadata.get("num_frames", zc1.shape[-1]),
        }
    except Exception:
        return None


def _save_cached(
    cache_dir: Path,
    item_id: str,
    zc1: torch.Tensor,
    phone_ids: torch.Tensor,
    speaker_id: str,
) -> None:
    """Save extracted latents and phone IDs to cache."""
    d = _cache_path(cache_dir, item_id)
    d.mkdir(parents=True, exist_ok=True)

    torch.save(zc1.cpu(), d / "latents.pt")
    torch.save(phone_ids.cpu(), d / "phone_ids.pt")

    metadata = {
        "speaker_id": speaker_id,
        "item_id": item_id,
        "num_frames": int(zc1.shape[-1]),
        "phone_frames": int(phone_ids.shape[-1]),
    }
    with open(d / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NativePriorDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that pre-extracts and caches FACodec latents and
    phoneme IDs for each utterance.

    The dataset is config-driven.  Pass a configuration dict or the path to
    a YAML file.

    Minimal example::

        cfg = yaml.safe_load(open("configs/phase1/overfit.yaml"))
        ds = NativePriorDataset(cfg, items=my_items)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=collate_fn, shuffle=True,
        )

    Args:
        config: Dict with dataset configuration, or path to a YAML file.
        items: List of ``(wav_path, speaker_id, transcript)`` tuples.
               These are the utterances to load.  If *None*, the dataset
               is empty (useful for unit tests).
    """

    def __init__(
        self,
        config: Dict[str, Any] | str,
        items: Optional[List[Tuple[str, str, str]]] = None,
    ):
        # ------------------------------------------------------------------ #
        # Load config
        # ------------------------------------------------------------------ #
        if isinstance(config, (str, os.PathLike)):
            config = _load_config(Path(config))

        ds_cfg = config.get("dataset", {})
        paths_cfg = config.get("paths", {})

        self.sample_rate: int = int(ds_cfg.get("sample_rate", 24000))
        self.codec_hop: int = int(ds_cfg.get("codec_hop", 300))
        self.codec_fps: int = int(ds_cfg.get("codec_fps", 80))
        self.facodec_device: str = ds_cfg.get("facodec_device", "cpu")
        self.normalize_latents: bool = bool(
            ds_cfg.get("normalize_latents", False)
        )
        self.overfit_num: Optional[int] = config.get("overfit", {}).get(
            "num_utterances"
        )

        self.facodec_ckpt: str = paths_cfg.get(
            "facodec_checkpoint", "Plachta/FAcodec"
        )
        self.cache_dir: Path = Path(
            paths_cfg.get("cache_dir", "./cache/facodec_phones")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Models (initialised lazily)
        # ------------------------------------------------------------------ #
        self._facodec: Optional[FACodecAdapter] = None
        self._phoneme: Optional[PhonemePipeline] = None

        # ------------------------------------------------------------------ #
        # Items
        # ------------------------------------------------------------------ #
        self._items: List[Tuple[str, str, str]] = items or []
        if self.overfit_num is not None and len(self._items) > self.overfit_num:
            self._items = self._items[: self.overfit_num]

        # ------------------------------------------------------------------ #
        # Pre-extract / load cache
        # ------------------------------------------------------------------ #
        self._index: List[Dict[str, Any]] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_facodec(self) -> FACodecAdapter:
        if self._facodec is None:
            self._facodec = FACodecAdapter(
                device=self.facodec_device,
                facodec_ckpt=self.facodec_ckpt,
            )
            self._facodec.freeze()
        return self._facodec

    def _get_phoneme(self) -> PhonemePipeline:
        if self._phoneme is None:
            self._phoneme = PhonemePipeline(
                device=self.facodec_device,
                sample_rate=self.sample_rate,
            )
        return self._phoneme

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Extract or load cached latents and phone IDs for all items."""
        import torchaudio

        facodec = self._get_facodec()
        phoneme = self._get_phoneme()

        for wav_path, speaker_id, transcript in self._items:
            item_id = f"{Path(wav_path).stem}__{speaker_id}"

            cached = _load_cached(self.cache_dir, item_id)
            if cached is not None:
                zc1 = cached["zc1"]
                phone_ids = cached["phone_ids"]
            else:
                # Load waveform
                wav, sr = torchaudio.load(wav_path)
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(
                        wav, orig_freq=sr, new_freq=self.sample_rate
                    )
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.to(torch.float32)

                # Extract latents
                with torch.no_grad():
                    latents = facodec.encode(wav)
                # latents.content_zc1: [B=1, 1, T]
                zc1 = latents.content_zc1.squeeze(0).squeeze(0)  # [T]

                # Extract phone IDs
                phone_ids = phoneme(transcript, wav).squeeze(0)  # [T]

                # Cache
                _save_cached(
                    self.cache_dir, item_id,
                    zc1.unsqueeze(0).unsqueeze(0),  # [1, 1, T]
                    phone_ids.unsqueeze(0),          # [1, T]
                    speaker_id,
                )

            # Normalize latents if requested
            if self.normalize_latents and zc1.numel() > 1:
                zc1 = (zc1 - zc1.mean()) / (zc1.std() + 1e-8)

            # Valid frame mask: all frames are valid by default.
            # In future, this could mask out low-energy / silence frames.
            valid_mask = torch.ones(zc1.shape[-1], dtype=torch.bool)

            self._index.append(
                {
                    "zc1": zc1,            # [T]  int64
                    "phone_ids": phone_ids,  # [T]  int64
                    "valid_mask": valid_mask,  # [T]  bool
                    "speaker_id": speaker_id,
                    "item_id": item_id,
                }
            )

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        entry = self._index[idx]
        return (
            entry["zc1"],           # [T] int64
            entry["phone_ids"],     # [T] int64
            entry["valid_mask"],    # [T] bool
            entry["speaker_id"],    # str
            entry["item_id"],       # str
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def speaker_ids(self) -> List[str]:
        """Return sorted list of unique speaker IDs in the dataset."""
        return sorted(set(e["speaker_id"] for e in self._index))

    def num_speakers(self) -> int:
        return len(self.speaker_ids())


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]],
    pad_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Collate a variable-length batch into padded tensors.

    Args:
        batch: List of ``(zc1, phone_ids, valid_mask, speaker_id, item_id)``.
        pad_value: Pad token ID (used for both zc1 and phone_ids).

    Returns:
        zc1        : [B, 1, T_max]  int64  padded zc1 codebook indices
        phone_ids  : [B, T_max]     int64  padded phoneme IDs
        valid_mask : [B, T_max]     bool   padded validity mask
        speaker_ids: List[str]               speaker strings
        item_ids   : List[str]               item identifier strings
    """
    zc1_list, phone_list, mask_list, spk_list, id_list = zip(*batch)

    T_max = max(z.shape[-1] for z in zc1_list)

    # Pad zc1: [1, T_i] -> [1, T_max] -> stack -> [B, 1, T_max]
    zc1_padded = torch.stack(
        [F.pad(z.unsqueeze(0), (0, T_max - z.shape[-1]), value=pad_value)
         for z in zc1_list]
    )  # [B, 1, T_max]

    # Pad phone_ids: [T_i] -> [T_max] -> stack -> [B, T_max]
    phone_padded = torch.stack(
        [F.pad(p, (0, T_max - p.shape[-1]), value=pad_value)
         for p in phone_list]
    )  # [B, T_max]

    # Pad valid_mask: [T_i] -> [T_max] -> stack -> [B, T_max]
    mask_padded = torch.stack(
        [F.pad(m, (0, T_max - m.shape[-1]), value=False)
         for m in mask_list]
    )  # [B, T_max]

    return zc1_padded, phone_padded, mask_padded, list(spk_list), list(id_list)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dict."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load AccentEdge config files. "
            "Install with: pip install pyyaml"
        ) from exc

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")

    return cfg
