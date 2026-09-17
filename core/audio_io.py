"""
audio_io.py — Single audio I/O path for the whole project.

Replaces the three near-identical readers that used to live in metrics.py and
steganalysis_extended.py (`_read_robust`, `_read_samples`, `_read_samples_robust`),
each of which swallowed different errors and returned a different dtype.

Everything that touches a WAV goes through here so that the benchmark, the
steganalysis battery and the encoders all see byte-identical sample arrays.
"""

import numpy as np
from scipy.io import wavfile


class UnreadableAudio(Exception):
    """Raised when neither scipy nor soundfile can decode the file."""


def read_wav(path: str):
    """
    Reads a WAV file, falling back to libsndfile for headers scipy rejects.

    scipy enforces `nAvgBytesPerSec == sample_rate * block_align`, which some
    encoders get wrong (dataset/ambient/nature8.wav is one). The audio itself is
    perfectly valid, so we fall back rather than dropping the file — the original
    benchmark silently skipped it via a bare `except: continue`, which is why the
    paper reports 40 files and the results CSV contains 43.

    Returns:
        (sample_rate, samples) — samples keeps its native shape and dtype.
    """
    try:
        return wavfile.read(path)
    except ValueError:
        pass

    try:
        import soundfile as sf
    except ImportError as exc:
        raise UnreadableAudio(
            f"{path}: scipy rejected the header and soundfile is not installed"
        ) from exc

    try:
        samples, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    except Exception as exc:
        raise UnreadableAudio(f"{path}: {exc}") from exc

    return sample_rate, samples


def read_mono(path: str) -> np.ndarray:
    """
    Returns the embedding channel (left) as a 1-D array in its native integer dtype.
    """
    _, samples = read_wav(path)
    if samples.ndim == 2:
        samples = samples[:, 0]
    return samples


def require_integer_pcm(samples: np.ndarray, path: str = "") -> np.ndarray:
    """
    Guards the encoders against float WAVs.

    LSB substitution is only meaningful on integer PCM. scipy returns float32 in
    [-1, 1] for IEEE-float WAVs, and the original encoder cast that straight to
    int32 — silently zeroing the entire signal. Fail loudly instead.
    """
    if np.issubdtype(samples.dtype, np.floating):
        raise ValueError(
            f"{path or 'audio'}: float WAV ({samples.dtype}) is not supported. "
            "LSB embedding requires integer PCM (int16/int32/uint8)."
        )
    return samples
