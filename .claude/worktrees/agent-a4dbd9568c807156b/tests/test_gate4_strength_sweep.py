"""Tests for scripts/gate4_strength_sweep.py"""
import json
import types
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Ensure mock audiotools is in place (mirrors gate4 script)
# ---------------------------------------------------------------------------
warnings.simplefilter("ignore")


def _make_mock(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    return m


if "audiotools" not in __import__("sys").modules:
    mock_audio = _make_mock("audiotools")
    mock_ml = _make_mock("audiotools.ml")
    mock_ml.BaseModel = type("BaseModel", (), {"INTERN": [], "EXTERN": []})
    mock_audio.ml = mock_ml
    mock_audio.AudioSignal = type("AudioSignal", (), {})
    mock_audio.STFTParams = type("STFTParams", (), {})
    mock_core = _make_mock("audiotools.core")
    mock_core.util = _make_mock("audiotools.core.util")
    __import__("sys").modules["audiotools"] = mock_audio
    __import__("sys").modules["audiotools.ml"] = mock_ml
    __import__("sys").modules["audiotools.core"] = mock_core
    __import__("sys").modules["audiotools.core.util"] = mock_core.util


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_wav(duration_sec=1.0, sr=24000):
    """Return a (1, N) float32 torch tensor of random audio."""
    n = int(duration_sec * sr)
    return torch.randn(1, n, dtype=torch.float32)


def _make_converter():
    """Return a mock AccentConverter whose convert() returns plausible outputs."""
    converter = MagicMock()
    # convert(wav, transcript, strength=...) -> (wav_out, meta)
    def _convert(wav, transcript, strength=0.0):
        # Return a slightly distorted version of the input
        out = wav.clone()
        # Add a small strength-dependent offset so identity shift grows
        out = out + strength * 0.01
        meta = {"strength": strength}
        return out, meta

    converter.convert.side_effect = _convert
    return converter


def _make_identity_eval():
    """Return a mock IdentityEvaluator."""
    eval_ = MagicMock()

    def _similarity(wav_a, wav_b, sr=24000):
        # Cosine similarity that decreases with strength difference
        # We control this via side_effect in individual tests
        return 0.9

    eval_.similarity.side_effect = _similarity
    return eval_


# ---------------------------------------------------------------------------
# Tests for evaluate_sample
# ---------------------------------------------------------------------------

class TestEvaluateSample:
    """Tests for the evaluate_sample function."""

    def test_output_structure(self):
        from scripts.gate4_strength_sweep import evaluate_sample

        strengths = [0.0, 0.5, 1.0]
        wav = _fake_wav(0.5)
        converter = _make_converter()
        identity_eval = _make_identity_eval()
        wer_eval = {"available": False, "model": None, "compute_wer": None}
        audio_dir = Path("/tmp/gate4_test_audio")
        audio_dir.mkdir(parents=True, exist_ok=True)

        result = evaluate_sample(
            idx=0,
            wav=wav,
            transcript="hello world",
            strengths=strengths,
            converter=converter,
            identity_eval=identity_eval,
            wer_eval=wer_eval,
            device=torch.device("cpu"),
            audio_dir=audio_dir,
            save_audio=False,
        )

        assert result["sample_idx"] == 0
        assert result["transcript"] == "hello world"
        assert "strengths" in result
        assert "monotonically_increasing" in result
        assert set(result["strengths"].keys()) == {"0.0", "0.5", "1.0"}

    def test_keys_per_strength(self):
        from scripts.gate4_strength_sweep import evaluate_sample

        strengths = [0.0, 1.0]
        wav = _fake_wav(0.5)
        converter = _make_converter()
        identity_eval = _make_identity_eval()
        wer_eval = {"available": False}
        audio_dir = Path("/tmp/gate4_test_audio2")
        audio_dir.mkdir(parents=True, exist_ok=True)

        result = evaluate_sample(
            idx=0,
            wav=wav,
            transcript="test",
            strengths=strengths,
            converter=converter,
            identity_eval=identity_eval,
            wer_eval=wer_eval,
            device=torch.device("cpu"),
            audio_dir=audio_dir,
            save_audio=False,
        )

        for s in strengths:
            entry = result["strengths"][str(s)]
            assert "mel_l1" in entry
            assert "identity_shift" in entry
            assert "identity_similarity" in entry
            assert "wer" in entry
            assert "metadata" in entry

    def test_identity_shift_range(self):
        from scripts.gate4_strength_sweep import evaluate_sample

        strengths = [0.0, 0.5, 1.0]
        wav = _fake_wav(0.5)
        converter = _make_converter()
        identity_eval = _make_identity_eval()
        identity_eval.similarity.return_value = 0.95
        wer_eval = {"available": False}
        audio_dir = Path("/tmp/gate4_test_audio3")
        audio_dir.mkdir(parents=True, exist_ok=True)

        result = evaluate_sample(
            idx=0,
            wav=wav,
            transcript="test",
            strengths=strengths,
            converter=converter,
            identity_eval=identity_eval,
            wer_eval=wer_eval,
            device=torch.device("cpu"),
            audio_dir=audio_dir,
            save_audio=False,
        )

        # identity_shift = 1 - similarity, so should be 0.05 when sim=0.95
        for s in strengths:
            shift = result["strengths"][str(s)]["identity_shift"]
            assert 0.0 <= shift <= 1.0
            # similarity + identity_shift == 1.0
            sim = result["strengths"][str(s)]["identity_similarity"]
            assert abs((sim + shift) - 1.0) < 1e-4

    def test_monotonic_increasing_flag(self):
        from scripts.gate4_strength_sweep import evaluate_sample

        strengths = [0.0, 0.5, 1.0]
        wav = _fake_wav(0.5)
        converter = _make_converter()
        identity_eval = _make_identity_eval()
        wer_eval = {"available": False}
        audio_dir = Path("/tmp/gate4_test_audio4")
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Make similarity decrease with strength so identity shift increases
        sim_values = [0.99, 0.80, 0.60]
        identity_eval.similarity.side_effect = [
            (wav.cpu().numpy().astype(np.float32).squeeze(),
             (wav.clone() + s * 0.01).cpu().numpy().astype(np.float32).squeeze())
            for s in strengths
        ]

        # Override side_effect to return controlled values
        def _sim(wav_a, wav_b, sr=24000):
            idx = strengths.index(round((1.0 - wav_b.numpy().std()) / 0.01, 1)) if wav_b.numpy().std() > 0 else 0
            return sim_values[min(idx, len(sim_values) - 1)]

        identity_eval.similarity.side_effect = _sim

        result = evaluate_sample(
            idx=0,
            wav=wav,
            transcript="test",
            strengths=strengths,
            converter=converter,
            identity_eval=identity_eval,
            wer_eval=wer_eval,
            device=torch.device("cpu"),
            audio_dir=audio_dir,
            save_audio=False,
        )

        # With decreasing similarity, shifts should be increasing
        shifts = [result["strengths"][str(s)]["identity_shift"] for s in strengths]
        assert result["monotonically_increasing"] is True
        for i in range(len(shifts) - 1):
            assert shifts[i] <= shifts[i + 1] + 1e-6

    def test_audio_saved_for_key_strengths(self):
        from scripts.gate4_strength_sweep import evaluate_sample

        strengths = [0.0, 0.5, 1.0]
        wav = _fake_wav(0.5)
        converter = _make_converter()
        identity_eval = _make_identity_eval()
        wer_eval = {"available": False}
        audio_dir = Path("/tmp/gate4_test_audio5")
        audio_dir.mkdir(parents=True, exist_ok=True)

        with patch("soundfile.write") as mock_sf:
            result = evaluate_sample(
                idx=2,
                wav=wav,
                transcript="test",
                strengths=strengths,
                converter=converter,
                identity_eval=identity_eval,
                wer_eval=wer_eval,
                device=torch.device("cpu"),
                audio_dir=audio_dir,
                save_audio=True,
            )

        # soundfile.write should be called for each strength (all are key strengths here)
        assert mock_sf.call_count == len(strengths)

        # audio_path should be in results
        for s in strengths:
            assert "audio_path" in result["strengths"][str(s)]

    def test_no_audio_when_save_audio_false(self):
        from scripts.gate4_strength_sweep import evaluate_sample

        strengths = [0.0, 1.0]
        wav = _fake_wav(0.5)
        converter = _make_converter()
        identity_eval = _make_identity_eval()
        wer_eval = {"available": False}
        audio_dir = Path("/tmp/gate4_test_audio6")
        audio_dir.mkdir(parents=True, exist_ok=True)

        with patch("soundfile.write") as mock_sf:
            result = evaluate_sample(
                idx=0,
                wav=wav,
                transcript="test",
                strengths=strengths,
                converter=converter,
                identity_eval=identity_eval,
                wer_eval=wer_eval,
                device=torch.device("cpu"),
                audio_dir=audio_dir,
                save_audio=False,
            )

        mock_sf.assert_not_called()
        for s in strengths:
            assert "audio_path" not in result["strengths"][str(s)]

    def test_wer_computed_when_available(self):
        from scripts.gate4_strength_sweep import evaluate_sample

        strengths = [0.0]
        wav = _fake_wav(0.5)
        converter = _make_converter()
        identity_eval = _make_identity_eval()
        wer_eval = {
            "available": True,
            "compute_wer": lambda ref, hyp: 0.0,
        }
        audio_dir = Path("/tmp/gate4_test_audio7")
        audio_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "scripts.gate4_strength_sweep.transcribe_audio",
            return_value="hello world",
        ):
            result = evaluate_sample(
                idx=0,
                wav=wav,
                transcript="hello world",
                strengths=strengths,
                converter=converter,
                identity_eval=identity_eval,
                wer_eval=wer_eval,
                device=torch.device("cpu"),
                audio_dir=audio_dir,
                save_audio=False,
            )

        # WER should be 0.0 since ref == hyp
        assert result["strengths"]["0.0"]["wer"] == 0.0


# ---------------------------------------------------------------------------
# Tests for evaluate_gate
# ---------------------------------------------------------------------------

class TestEvaluateGate:
    """Tests for the evaluate_gate function."""

    def _make_results(self, shifts_per_strength, mel_l1s_per_strength):
        """Build a fake all_results list for evaluate_gate."""
        strengths = list(shifts_per_strength.keys())
        n_samples = len(next(iter(shifts_per_strength.values())))
        results = []
        for i in range(n_samples):
            entry = {
                "sample_idx": i,
                "strengths": {},
                "monotonically_increasing": True,
            }
            for s in strengths:
                entry["strengths"][str(s)] = {
                    "identity_shift": shifts_per_strength[s][i],
                    "mel_l1": mel_l1s_per_strength[s][i],
                    "wer": None,
                }
            results.append(entry)
        return results

    def test_overall_pass_all_criteria_met(self):
        from scripts.gate4_strength_sweep import evaluate_gate

        strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
        shifts = {s: [0.0, 0.0] for s in strengths}
        mel_l1s = {s: [0.1, 0.15] for s in strengths}
        # Make strength=1.0 have high shift
        shifts[1.0] = [0.20, 0.22]
        # Make monotonic
        for i in range(2):
            shifts[0.0][i] = 0.0
            shifts[0.25][i] = 0.05
            shifts[0.5][i] = 0.10
            shifts[0.75][i] = 0.15
            shifts[1.0][i] = 0.20 + i * 0.02

        all_results = self._make_results(shifts, mel_l1s)
        gate = evaluate_gate(all_results, strengths)

        assert gate["overall_pass"] is True
        assert gate["n_samples"] == 2

    def test_overall_fail_low_id_shift_at_strength_1(self):
        from scripts.gate4_strength_sweep import evaluate_gate

        strengths = [0.0, 1.0]
        shifts = {0.0: [0.0, 0.0], 1.0: [0.10, 0.12]}  # Below 0.15 threshold
        mel_l1s = {0.0: [0.1, 0.1], 1.0: [0.2, 0.2]}

        all_results = self._make_results(shifts, mel_l1s)
        gate = evaluate_gate(all_results, strengths)

        assert gate["overall_pass"] is False

    def test_overall_fail_high_id_shift_at_strength_0(self):
        from scripts.gate4_strength_sweep import evaluate_gate

        strengths = [0.0, 1.0]
        shifts = {0.0: [0.10, 0.12], 1.0: [0.20, 0.22]}  # 0.0 shift > 0.05
        mel_l1s = {0.0: [0.1, 0.1], 1.0: [0.2, 0.2]}

        all_results = self._make_results(shifts, mel_l1s)
        gate = evaluate_gate(all_results, strengths)

        assert gate["overall_pass"] is False

    def test_overall_fail_high_mel_l1(self):
        from scripts.gate4_strength_sweep import evaluate_gate

        strengths = [0.0, 1.0]
        shifts = {0.0: [0.0, 0.0], 1.0: [0.20, 0.22]}
        mel_l1s = {0.0: [0.1, 0.1], 1.0: [0.6, 0.6]}  # Above 0.5 threshold

        all_results = self._make_results(shifts, mel_l1s)
        gate = evaluate_gate(all_results, strengths)

        assert gate["overall_pass"] is False

    def test_strength_stats_aggregation(self):
        from scripts.gate4_strength_sweep import evaluate_gate

        strengths = [0.0, 1.0]
        shifts = {0.0: [0.0, 0.01], 1.0: [0.20, 0.22]}
        mel_l1s = {0.0: [0.10, 0.12], 1.0: [0.20, 0.22]}

        all_results = self._make_results(shifts, mel_l1s)
        gate = evaluate_gate(all_results, strengths)

        stats_0 = gate["strength_stats"]["0.0"]
        assert abs(stats_0["mean_identity_shift"] - 0.005) < 1e-6
        assert abs(stats_0["mean_mel_l1"] - 0.11) < 1e-6

        stats_1 = gate["strength_stats"]["1.0"]
        assert abs(stats_1["mean_identity_shift"] - 0.21) < 1e-6
        assert abs(stats_1["mean_mel_l1"] - 0.21) < 1e-6


# ---------------------------------------------------------------------------
# Tests for build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    """Tests for the CLI argument parser."""

    def test_defaults(self):
        from scripts.gate4_strength_sweep import build_parser

        parser = build_parser()
        args = parser.parse_args([])

        assert args.device == "cpu"
        assert args.n_samples == 5
        assert args.output_dir == "artifacts/gate4"
        assert args.strengths == "0.0,0.25,0.5,0.75,1.0"
        assert args.checkpoint is None
        assert args.wer_model == "large-v3-turbo"

    def test_custom_args(self):
        from scripts.gate4_strength_sweep import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--device", "cuda",
            "--n-samples", "10",
            "--output-dir", "/tmp/test",
            "--strengths", "0.0,0.5,1.0",
            "--checkpoint", "/tmp/ckpt.pt",
            "--wer-model", "tiny",
        ])

        assert args.device == "cuda"
        assert args.n_samples == 10
        assert args.output_dir == "/tmp/test"
        assert args.strengths == "0.0,0.5,1.0"
        assert args.checkpoint == "/tmp/ckpt.pt"
        assert args.wer_model == "tiny"


# ---------------------------------------------------------------------------
# Tests for import_core
# ---------------------------------------------------------------------------

class TestImportCore:
    """Tests for the import_core function."""

    def test_returns_all_modules(self):
        from scripts.gate4_strength_sweep import import_core

        mods = import_core()
        expected_keys = {
            "torch", "np",
            "FACodecAdapter", "AccentConverter", "PhonemePipeline",
            "DenoisingTransformerModel", "ZC2Recomputer",
            "IdentityEvaluator", "mel_l1", "L2ArcticDataset",
        }
        assert set(mods.keys()) == expected_keys

    def test_modules_are_callable(self):
        from scripts.gate4_strength_sweep import import_core

        mods = import_core()
        assert callable(mods["FACodecAdapter"])
        assert callable(mods["AccentConverter"])
        assert callable(mods["PhonemePipeline"])
        assert callable(mods["IdentityEvaluator"])
        assert callable(mods["mel_l1"])
        assert callable(mods["L2ArcticDataset"])


# ---------------------------------------------------------------------------
# Tests for create_wer_evaluator
# ---------------------------------------------------------------------------

class TestCreateWEREvaluator:
    """Tests for the WER evaluator factory."""

    def test_returns_dict_with_keys(self):
        from scripts.gate4_strength_sweep import create_wer_evaluator

        result = create_wer_evaluator.__wrapped__("tiny")
        assert "available" in result
        assert "model" in result
        assert "compute_wer" in result

    def test_graceful_fallback_when_no_faster_whisper(self):
        from scripts.gate4_strength_sweep import create_wer_evaluator

        with patch.dict(__import__("sys").modules, {"faster_whisper": None}):
            result = create_wer_evaluator.__wrapped__("tiny")

        assert result["available"] is False
        assert result["model"] is None
        assert result["compute_wer"] is None
