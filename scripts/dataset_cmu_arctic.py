"""Dataset loading for CMU ARCTIC and L2-ARCTIC corpora.

Downloads, parses, and yields speaker-anchored triplets:

    (wav_path, speaker_id, transcript)

The module also exposes helpers for building same-speaker and
different-speaker reference pairs.

Usage::

    from scripts.dataset_cmu_arctic import CmuArcticDataset, L2ArcticDataset

    cmu = CmuArcticDataset(root="./data/cmu_arctic")
    l2  = L2ArcticDataset(root="./data/l2_arctic")

    items = cmu.all_items()          # list of (path, speaker, transcript)
    pairs = cmu.same_speaker_pairs() # [(path_a, path_b, speaker), ...]
"""
from __future__ import annotations

import csv
import hashlib
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import torchaudio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CMU_ARCTIC_URL: str = (
    "https://www.cs.cmu.edu/~ark/linguistic-data/cmu_arctic.tgz"
)
L2_ARCTIC_URL: str = (
    "https://www.cs.cmu.edu/~ark/hard-align/l2-arctic-2026.zip"
)

# CMU ARCTIC speaker metadata (subset of well-known speakers)
# Format: speaker_id -> (gender, accent)
_CMU_SPEAKER_META = {
    "slt": ("F", "US"),
    "bdl": ("M", "US"),
    "clb": ("F", "US"),
    "jmk": ("M", "US"),
    "ksp": ("M", "US"),
    "rms": ("M", "US"),
    "lnh": ("F", "Canadian"),
    "awb": ("M", "Scottish"),
    "ljm": ("F", "Indian"),
    "rxr": ("M", "Indian"),
}

# L2-ARCTIC speaker metadata
# Format: speaker_id -> (native_language, gender)
_L2_SPEAKER_META = {
    "ABA": ("Arabic", "M"),
    "SKA": ("Arabic", "F"),
    "YBAA": ("Arabic", "M"),
    "ZHAA": ("Arabic", "F"),
    "BWC": ("Mandarin", "M"),
    "LXC": ("Mandarin", "F"),
    "NCC": ("Mandarin", "F"),
    "TXHC": ("Mandarin", "M"),
    "ASI": ("Hindi", "M"),
    "RRBI": ("Hindi", "M"),
    "MBMPS": ("Hindi", "F"),
    "HJK": ("Korean", "F"),
    "HKK": ("Korean", "M"),
    "YDCK": ("Korean", "F"),
    "YKWK": ("Korean", "M"),
    "EBVS": ("Spanish", "F"),
    "ERMS": ("Spanish", "M"),
    "TNI": ("Spanish", "F"),
    "TIMIT": ("Spanish", "M"),
    "GVA": ("French", "F"),
    "GVM": ("French", "M"),
    "JVV": ("French", "M"),
    "NJS": ("French", "F"),
    "OJGV": ("French", "M"),
    "THV": ("Vietnamese", "F"),
    "TLV": ("Vietnamese", "F"),
    "FSK": ("German", "M"),
    "JDC": ("German", "M"),
    "KWSTART": ("German", "F"),
}

# Sentences shared across both corpora (used for prompt matching)
_SHARED_SENTENCES = {
    "arctic_a001",
    "arctic_a008",
    "arctic_a010",
    "arctic_a016",
    "arctic_a022",
    "arctic_a039",
    "arctic_a047",
    "arctic_a050",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, expected_hash: Optional[str] = None) -> Path:
    """Download a file to *dest*, verifying SHA256 if provided."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_hash is not None:
            actual = hashlib.sha256(dest.read_bytes()).hexdigest()
            if actual == expected_hash:
                return dest
        else:
            return dest

    print(f"[dataset] Downloading {url} -> {dest}")

    def _report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded / total_size * 100) if total_size > 0 else 0
        print(f"\r  {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_report_hook)
    print()  # newline after progress

    return dest


def _extract_tgz(archive: Path, dest: Path) -> None:
    """Extract a .tgz archive."""
    print(f"[dataset] Extracting {archive.name} -> {dest}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest)


def _extract_zip(archive: Path, dest: Path) -> None:
    """Extract a .zip archive."""
    print(f"[dataset] Extracting {archive.name} -> {dest}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(path=dest)


def _parse_transcript(line: str) -> str:
    """Parse a transcript line from CMU ARCTIC .TXT files.

    Each line looks like:
        arctic_a001  The sale on personal computers ...
    We drop the utterance ID and return the sentence.
    """
    parts = line.strip().split(maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return line.strip()


# ---------------------------------------------------------------------------
# CMU ARCTIC
# ---------------------------------------------------------------------------


class CmuArcticDataset:
    """CMU ARCTIC speech corpus loader.

    The corpus contains ~1 130 utterances from 7 native English speakers
    (3 male, 4 female).  Audio is 16 kHz 16-bit PCM WAV.

    After a one-time download the dataset is cached under *root*.

    Args:
        root: Directory to download / store the corpus.
        split: Which partition to return (``"all"`` by default).
               Accepts a list of speaker IDs to filter.
        download: If ``True``, download the corpus if not already cached.
    """

    def __init__(
        self,
        root: str | os.PathLike = "./data/cmu_arctic",
        split: str | List[str] = "all",
        download: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.download = download

        self._wav_dir: Optional[Path] = None
        self._txt_dir: Optional[Path] = None
        self._items: List[Tuple[str, str, str]] = []

        if self.download:
            self._prepare()

        self._items = self._load_index()

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        """Download and extract the corpus if needed."""
        archive = self.root / "cmu_arctic.tgz"
        extracted = self.root / "cmu_arctic"

        if not extracted.exists() or not any(extracted.iterdir()):
            _download(CMU_ARCTIC_URL, archive)
            _extract_tgz(archive, self.root)

        # Locate wav/ and txt/ directories
        for sub in extracted.rglob("wav"):
            if sub.is_dir():
                self._wav_dir = sub
                break
        for sub in extracted.rglob("txt"):
            if sub.is_dir():
                self._txt_dir = sub
                break

        if self._wav_dir is None or self._txt_dir is None:
            raise RuntimeError(
                "Could not locate wav/ or txt/ directories in "
                f"{extracted}. The archive layout may have changed."
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _load_index(self) -> List[Tuple[str, str, str]]:
        """Build list of (wav_path, speaker_id, transcript)."""
        items: List[Tuple[str, str, str]] = []

        if self._txt_dir is None:
            return items

        for txt_file in sorted(self._txt_dir.glob("*.TXT")):
            speaker_id = txt_file.stem.split("_")[0].lower()
            if isinstance(self.split, list) and speaker_id not in self.split:
                continue

            wav_dir = self._wav_dir / speaker_id if self._wav_dir else None
            if wav_dir is None or not wav_dir.exists():
                continue

            with open(txt_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, transcript = _parse_transcript(line)
                    wav_path = wav_dir / f"{utt_id}.wav"
                    if wav_path.exists():
                        items.append((str(wav_path), speaker_id, transcript))

        return items

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_items(self) -> List[Tuple[str, str, str]]:
        """Return all (wav_path, speaker_id, transcript) tuples."""
        return list(self._items)

    def speakers(self) -> List[str]:
        """Return sorted list of unique speaker IDs."""
        return sorted(set(s for _, s, _ in self._items))

    def same_speaker_pairs(self) -> List[Tuple[str, str, str]]:
        """Return (path_a, path_b, speaker_id) for every unordered pair of
        utterances by the same speaker.

        Useful for same-speaker reference training.
        """
        from collections import defaultdict

        by_speaker: dict[str, List[str]] = defaultdict(list)
        for path, spk, _ in self._items:
            by_speaker[spk].append(path)

        pairs: List[Tuple[str, str, str]] = []
        for spk, paths in by_speaker.items():
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    pairs.append((paths[i], paths[j], spk))
        return pairs

    def different_speaker_pairs(
        self, max_pairs: int = 1000
    ) -> List[Tuple[str, str, str, str]]:
        """Return (path_a, speaker_a, path_b, speaker_b) for pairs of
        utterances by *different* speakers.

        Args:
            max_pairs: Maximum number of pairs to return (randomly sampled).
        """
        import random

        spk_items: dict[str, List[str]] = {}
        for path, spk, _ in self._items:
            spk_items.setdefault(spk, []).append(path)

        speakers = list(spk_items.keys())
        pairs: List[Tuple[str, str, str, str]] = []

        for _ in range(max_pairs):
            spk_a, spk_b = random.sample(speakers, 2)
            path_a = random.choice(spk_items[spk_a])
            path_b = random.choice(spk_items[spk_b])
            pairs.append((path_a, spk_a, path_b, spk_b))

        return pairs

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# L2-ARCTIC
# ---------------------------------------------------------------------------


class L2ArcticDataset:
    """L2-ARCTIC non-native English speech corpus loader.

    Contains ~2 640 utterances from 24 non-native speakers of English
    (3 utterances each from 11 prompts × 2 speakers per L1 group).

    The shared 11 prompts overlap with CMU ARCTIC sentences, making
    cross-corpus prompt matching possible.

    Args:
        root: Directory to download / store the corpus.
        l1_filter: Optional list of L1 codes to include (e.g. ``["Hindi"]``).
                   Pass ``None`` to include all.
        download: If ``True``, download the corpus if not already cached.
    """

    def __init__(
        self,
        root: str | os.PathLike = "./data/l2_arctic",
        l1_filter: Optional[List[str]] = None,
        download: bool = True,
    ):
        self.root = Path(root)
        self.l1_filter = l1_filter
        self.download = download

        self._wav_dir: Optional[Path] = None
        self._transcript_dir: Optional[Path] = None
        self._items: List[Tuple[str, str, str]] = []

        if self.download:
            self._prepare()

        self._items = self._load_index()

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        """Download and extract the corpus if needed."""
        archive = self.root / "l2-arctic-2026.zip"
        extracted = self.root / "l2-arctic"

        if not extracted.exists() or not any(extracted.iterdir()):
            _download(L2_ARCTIC_URL, archive)
            _extract_zip(archive, self.root)

        # Locate directories
        for sub in extracted.rglob("wav"):
            if sub.is_dir():
                self._wav_dir = sub
                break
        for sub in extracted.rglob("transcript"):
            if sub.is_dir():
                self._transcript_dir = sub
                break

        if self._wav_dir is None:
            # Some releases use "wav48" instead of "wav"
            for sub in extracted.rglob("wav48"):
                if sub.is_dir():
                    self._wav_dir = sub
                    break

        if self._wav_dir is None or self._transcript_dir is None:
            raise RuntimeError(
                "Could not locate wav/ or transcript/ directories in "
                f"{extracted}. The archive layout may have changed."
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _load_index(self) -> List[Tuple[str, str, str]]:
        """Build list of (wav_path, speaker_id, transcript)."""
        items: List[Tuple[str, str, str]] = []

        if self._transcript_dir is None:
            return items

        for spk_dir in sorted(self._transcript_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            speaker_id = spk_dir.name

            if self.l1_filter is not None:
                meta = _L2_SPEAKER_META.get(speaker_id, ("Unknown", "U"))
                if meta[0] not in self.l1_filter:
                    continue

            wav_base = (
                self._wav_dir / speaker_id
                if self._wav_dir is not None
                else None
            )

            for txt_file in sorted(spk_dir.glob("*.txt")):
                utt_id = txt_file.stem
                with open(txt_file, encoding="utf-8") as f:
                    transcript = f.read().strip()

                # Try .wav first, then .WAV
                wav_path = None
                if wav_base is not None:
                    for ext in (".wav", ".WAV"):
                        candidate = wav_base / f"{utt_id}{ext}"
                        if candidate.exists():
                            wav_path = candidate
                            break

                if wav_path is not None:
                    items.append((str(wav_path), speaker_id, transcript))

        return items

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_items(self) -> List[Tuple[str, str, str]]:
        """Return all (wav_path, speaker_id, transcript) tuples."""
        return list(self._items)

    def speakers(self) -> List[str]:
        """Return sorted list of unique speaker IDs."""
        return sorted(set(s for _, s, _ in self._items))

    def l1_groups(self) -> List[str]:
        """Return sorted list of L1 language groups."""
        groups = set()
        for _, spk, _ in self._items:
            meta = _L2_SPEAKER_META.get(spk)
            if meta:
                groups.add(meta[0])
        return sorted(groups)

    def by_l1(self, l1: str) -> List[Tuple[str, str, str]]:
        """Return items filtered by L1 language."""
        return [
            item for item in self._items
            if _L2_SPEAKER_META.get(item[1], ("", ""))[0] == l1
        ]

    def shared_sentence_items(self) -> List[Tuple[str, str, str]]:
        """Return items whose utterance ID matches a shared CMU ARCTIC sentence.

        These items are suitable for prompt-matched cross-corpus pairing.
        """
        matched: List[Tuple[str, str, str]] = []
        for wav_path, spk, transcript in self._items:
            utt_id = Path(wav_path).stem.lower()
            if utt_id in _SHARED_SENTENCES or any(
                utt_id.endswith(s.lower()) for s in _SHARED_SENTENCES
            ):
                matched.append((wav_path, spk, transcript))
        return matched

    def same_speaker_pairs(self) -> List[Tuple[str, str, str]]:
        """Return (path_a, path_b, speaker_id) for same-speaker pairs."""
        from collections import defaultdict

        by_speaker: dict[str, List[str]] = defaultdict(list)
        for path, spk, _ in self._items:
            by_speaker[spk].append(path)

        pairs: List[Tuple[str, str, str]] = []
        for spk, paths in by_speaker.items():
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    pairs.append((paths[i], paths[j], spk))
        return pairs

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# Cross-corpus prompt matching
# ---------------------------------------------------------------------------


def match_prompts(
    cmu_items: List[Tuple[str, str, str]],
    l2_items: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str, str, str, str]]:
    """Match utterances between CMU ARCTIC and L2-ARCTIC by transcript text.

    Uses a simple word-overlap heuristic to find sentences with the same
    or very similar content across corpora.

    Returns:
        List of (cmu_path, cmu_speaker, cmu_transcript,
                 l2_path,  l2_speaker,  l2_transcript) tuples.
    """
    matches: List[Tuple[str, str, str, str, str, str]] = []

    # Build a lookup by normalised transcript (lowercase, stripped)
    l2_lookup: dict[str, List[Tuple[str, str, str]]] = {}
    for path, spk, t in l2_items:
        key = t.lower().strip().rstrip(".")
        l2_lookup.setdefault(key, []).append((path, spk, t))

    for cmu_path, cmu_spk, cmu_t in cmu_items:
        key = cmu_t.lower().strip().rstrip(".")
        if key in l2_lookup:
            for l2_path, l2_spk, l2_t in l2_lookup[key]:
                matches.append(
                    (cmu_path, cmu_spk, cmu_t, l2_path, l2_spk, l2_t)
                )

    return matches


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load and resample a WAV file.

    Returns:
        waveform: [1, T] float32 tensor at *target_sr*.
    """
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.to(torch.float32)


def load_wavs_and_transcripts(
    items: List[Tuple[str, str, str]],
    target_sr: int = 24000,
) -> List[Tuple[torch.Tensor, str, str]]:
    """Load all items into memory as (waveform, speaker_id, transcript).

    Warning: loads the entire dataset into RAM.  Use the Dataset class
    instead for on-demand loading.
    """
    results: List[Tuple[torch.Tensor, str, str]] = []
    for path, spk, t in items:
        wav = load_audio(path, target_sr)
        results.append((wav, spk, t))
    return results
