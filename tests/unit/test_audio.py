"""Tests for accentedge_benchmark.audio.io - load, save, resample, mono."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from accentedge_benchmark.audio.io import (
    load_audio, save_audio, resample_audio, ensure_mono,
)


@pytest.fixture
def tmp_wav(tmp_path):
    """Create a temporary mono WAV file."""
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), waveform, sr, subtype="PCM_16")
    return path, waveform, sr


class TestLoadAudio:
    def test_load_mono(self, tmp_wav):
        path, expected, sr = tmp_wav
        wf, actual_sr = load_audio(path)
        assert actual_sr == sr
        assert wf.dtype == np.float32
        np.testing.assert_array_almost_equal(wf, expected, decimal=4)

    def test_load_with_resample(self, tmp_wav):
        path, _, _ = tmp_wav
        wf, actual_sr = load_audio(path, sr=8000)
        assert actual_sr == 8000
        assert wf.dtype == np.float32
        assert len(wf) > 0

    def test_load_stereo_to_mono(self, tmp_path):
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wf = np.stack(
            [0.5 * np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 880 * t)],
            axis=1,
        ).astype(np.float32)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), wf, sr, subtype="PCM_16")
        loaded, _ = load_audio(path, mono=True)
        assert loaded.ndim == 1

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_audio(tmp_path / "nonexistent.wav")

    def test_load_preserves_float32(self, tmp_wav):
        path, _, _ = tmp_wav
        wf, _ = load_audio(path)
        assert wf.dtype == np.float32


class TestSaveAudio:
    def test_round_trip(self, tmp_path):
        sr = 16000
        original = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        fpath = tmp_path / "roundtrip.wav"
        save_audio(fpath, original, sr)
        loaded, _ = load_audio(fpath)
        np.testing.assert_array_almost_equal(loaded, original, decimal=4)

    def test_creates_parent_dirs(self, tmp_path):
        fpath = tmp_path / "deep" / "nested" / "out.wav"
        save_audio(fpath, np.zeros(100, dtype=np.float32), 16000)
        assert fpath.exists()

    def test_saves_float32(self, tmp_path):
        fpath = tmp_path / "f32.wav"
        save_audio(fpath, np.ones(100, dtype=np.float32), 16000)
        loaded, _ = load_audio(fpath)
        assert loaded.dtype == np.float32


class TestResampleAudio:
    def test_same_rate_no_change(self):
        wf = np.ones(100, dtype=np.float32)
        result = resample_audio(wf, 16000, 16000)
        np.testing.assert_array_equal(result, wf)

    def test_downsample(self):
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wf = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        resampled = resample_audio(wf, sr, 8000)
        assert resampled.dtype == np.float32
        assert len(resampled) > 0


class TestEnsureMono:
    def test_mono_unchanged(self):
        wf = np.zeros(100, dtype=np.float32)
        result = ensure_mono(wf)
        np.testing.assert_array_equal(result, wf)

    def test_stereo_converted(self):
        wf = np.stack(
            [np.ones(100), np.ones(100) * 2], axis=1
        ).astype(np.float32)
        result = ensure_mono(wf)
        expected = np.ones(100, dtype=np.float32) * 1.5
        np.testing.assert_array_almost_equal(result, expected)
