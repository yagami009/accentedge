"""Tests for accentedge_benchmark.audio.io."""

import numpy as np
import pytest
import soundfile as sf

from accentedge_benchmark.audio.io import (
    load_audio,
    save_audio,
    resample_audio,
    ensure_mono,
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


class TestSaveAudio:
    def test_save_and_reload(self, tmp_path):
        sr = 16000
        waveform = np.random.randn(sr).astype(np.float32) * 0.1
        out_path = tmp_path / "subdir" / "out.wav"
        save_audio(out_path, waveform, sr)
        assert out_path.exists()
        wf, actual_sr = sf.read(str(out_path))
        assert actual_sr == sr
        np.testing.assert_array_almost_equal(wf.astype(np.float32), waveform, decimal=4)


class TestResampleAudio:
    def test_same_rate(self):
        wf = np.zeros(100, dtype=np.float32)
        result = resample_audio(wf, 16000, 16000)
        np.testing.assert_array_equal(result, wf)

    def test_resample(self):
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wf = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        resampled = resample_audio(wf, sr, 8000)
        assert resampled.dtype == np.float32
        assert len(resampled) == len(wf) // 2


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
