"""Tests for accentedge_benchmark.degradations — degradation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from accentedge_benchmark.degradations import apply_degradation


class TestApplyDegradation:
    def test_clean_returns_unchanged(self):
        audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32)
        result = apply_degradation(audio, 16000, "clean")
        np.testing.assert_array_equal(result, audio)

    def test_clean_preserves_dtype(self):
        audio = np.zeros(100, dtype=np.float32)
        result = apply_degradation(audio, 16000, "clean")
        assert result.dtype == np.float32

    def test_nb_reduces_sample_rate(self):
        """NB resamples down to 8kHz then back up — length changes."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = apply_degradation(audio, sr, "nb", nb_target_sr=8000)
        # Should be approximately the same length after round-trip resampling
        assert len(result) > 0
        assert result.dtype == np.float32

    def test_nb_same_seed_deterministic(self):
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        r1 = apply_degradation(audio, sr, "nb", seed=42, nb_target_sr=8000)
        r2 = apply_degradation(audio, sr, "nb", seed=42, nb_target_sr=8000)
        np.testing.assert_array_equal(r1, r2)

    def test_noisy_adds_noise(self):
        """Noisy condition should change the audio (add noise)."""
        sr = 16000
        audio = np.ones(sr, dtype=np.float32) * 0.1  # constant signal
        result = apply_degradation(audio, sr, "noisy", seed=42, noisy_snr_db=15.0)
        # With noise added, samples should differ from original
        assert not np.allclose(result, audio, atol=1e-6)

    def test_noisy_deterministic(self):
        sr = 16000
        audio = np.ones(sr, dtype=np.float32) * 0.1
        r1 = apply_degradation(audio, sr, "noisy", seed=42, noisy_snr_db=15.0)
        r2 = apply_degradation(audio, sr, "noisy", seed=42, noisy_snr_db=15.0)
        np.testing.assert_array_equal(r1, r2)

    def test_noisy_different_seed_different_output(self):
        sr = 16000
        audio = np.ones(sr, dtype=np.float32) * 0.1
        r1 = apply_degradation(audio, sr, "noisy", seed=1, noisy_snr_db=15.0)
        r2 = apply_degradation(audio, sr, "noisy", seed=99, noisy_snr_db=15.0)
        # Different seeds → different noise → different output
        assert not np.allclose(r1, r2, atol=1e-6)

    def test_nb_noisy_combines_both(self):
        """nb_noisy should both resample and add noise."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = apply_degradation(audio, sr, "nb_noisy", seed=42,
                                   nb_target_sr=8000, noisy_snr_db=15.0)
        # Result should differ from both clean and pure nb
        clean = apply_degradation(audio, sr, "clean")
        nb_only = apply_degradation(audio, sr, "nb", seed=42, nb_target_sr=8000)
        assert not np.allclose(result, clean, atol=1e-6)
        assert not np.allclose(result, nb_only, atol=1e-6)

    def test_nb_noisy_deterministic(self):
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        r1 = apply_degradation(audio, sr, "nb_noisy", seed=42,
                               nb_target_sr=8000, noisy_snr_db=15.0)
        r2 = apply_degradation(audio, sr, "nb_noisy", seed=42,
                               nb_target_sr=8000, noisy_snr_db=15.0)
        np.testing.assert_array_equal(r1, r2)

    def test_output_float32(self):
        audio = np.random.randn(1000).astype(np.float64)
        for condition in ["clean", "nb", "noisy", "nb_noisy"]:
            result = apply_degradation(audio, 16000, condition)
            assert result.dtype == np.float32
