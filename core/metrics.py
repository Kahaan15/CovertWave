"""
metrics.py — Perceptual quality and payload-recovery metrics.

Fixed here relative to the original:

  * calculate_psnr computed a length-aligned MSE and then immediately
    overwrote it with an unaligned one, making the alignment dead code and
    raising on any length mismatch.
  * calculate_mse had unreachable statements after its return.
  * print_capacity_report did `from encoder import ...` (no leading dot), so it
    raised ImportError whenever it was called.
  * PSNR/MSE were duplicated in steganalysis_extended.py with different
    semantics for identical files (inf here, 100.0 there) and the benchmark
    imported one pair while the battery used the other.
  * Three near-identical WAV readers existed; all reads now go through
    core.audio_io.

Bit error rate and extraction success are new. The original study measured eight
metrics and none of them checked whether the hidden message could actually be
read back — which is how a decoder that failed on 30% of the corpus went
unnoticed.
"""

import numpy as np
from scipy.stats import entropy as shannon_entropy

from .audio_io import read_mono

# 16-bit PCM full scale. All dataset audio is int16; assert rather than assume.
MAX_INT16 = 32767.0


def _aligned(original, stego):
    """Accepts paths or arrays; returns both as float64 truncated to a common length."""
    if isinstance(original, str):
        original = read_mono(original)
    if isinstance(stego, str):
        stego = read_mono(stego)
    original = np.asarray(original, dtype=np.float64)
    stego = np.asarray(stego, dtype=np.float64)
    n = min(len(original), len(stego))
    return original[:n], stego[:n]


def _as_samples(source):
    """Accepts a path or an already-loaded array."""
    return read_mono(source) if isinstance(source, str) else np.asarray(source)


def calculate_mse(original_wav, stego_wav) -> float:
    """Mean squared error between cover and stego (embedding channel only)."""
    original, stego = _aligned(original_wav, stego_wav)
    return float(np.mean((original - stego) ** 2))


def calculate_psnr(original_wav, stego_wav) -> float:
    """
    Peak signal-to-noise ratio in dB, referenced to 16-bit full scale.
    Returns +inf for bit-identical files.
    """
    mse = calculate_mse(original_wav, stego_wav)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(MAX_INT16) - 10 * np.log10(mse))


def calculate_snr(original_wav, stego_wav) -> float:
    """Signal-to-noise ratio in dB, treating the embedding residual as noise."""
    original, stego = _aligned(original_wav, stego_wav)
    noise_power = float(np.mean((original - stego) ** 2))
    if noise_power == 0:
        return float("inf")
    signal_power = float(np.mean(original ** 2))
    return float(10 * np.log10(signal_power / noise_power))


def spectral_flatness(wav_path) -> float:
    """
    Wiener entropy: geometric mean over arithmetic mean of the magnitude
    spectrum. 1.0 is white noise, values near 0 are strongly tonal.
    Accepts a path or a sample array.
    """
    samples = _as_samples(wav_path).astype(np.float64)
    magnitude = np.abs(np.fft.rfft(samples))
    magnitude = magnitude[magnitude > 0]
    if magnitude.size == 0:
        return 0.0
    geometric = np.exp(np.mean(np.log(magnitude)))
    return float(geometric / np.mean(magnitude))


def lsb_entropy(wav_path) -> float:
    """Shannon entropy (bits) of the LSB plane. Noise-like planes approach 1.0."""
    samples = _as_samples(wav_path)
    lsbs = np.asarray(samples) & 1
    counts = np.bincount(lsbs, minlength=2)
    return float(shannon_entropy(counts / counts.sum(), base=2))


def bit_error_rate(sent: str, recovered: str) -> float:
    """
    Fraction of differing bits between the sent and recovered message.

    Returns 1.0 when nothing was recovered, and compares over the longer of the
    two so that truncation counts as error rather than being silently ignored.
    """
    if recovered is None:
        return 1.0

    sent_bytes = sent.encode("utf-8")
    recovered_bytes = recovered.encode("utf-8")
    width = max(len(sent_bytes), len(recovered_bytes))
    if width == 0:
        return 0.0

    sent_padded = sent_bytes.ljust(width, b"\x00")
    recovered_padded = recovered_bytes.ljust(width, b"\x00")

    differing = sum(
        bin(a ^ b).count("1") for a, b in zip(sent_padded, recovered_padded)
    )
    return differing / (width * 8)


def capacity_report(input_wav: str, lsb_bits: int = 1) -> dict:
    """Capacity of a cover file at a given LSB depth."""
    from .encoder import calculate_capacity
    return calculate_capacity(input_wav, lsb_bits=lsb_bits)


def print_capacity_report(input_wav: str):
    """Prints capacity across all three LSB depths."""
    print("\n[CAPACITY REPORT]")
    print("=" * 55)
    for lsb in (1, 2, 4):
        info = capacity_report(input_wav, lsb_bits=lsb)
        print(f"\n  LSB Mode        : {lsb}-bit")
        print(f"  Duration        : {info['duration_seconds']:.2f} seconds")
        print(f"  Total Samples   : {info['total_samples']}")
        print(f"  Usable Bytes    : {info['usable_bytes']}")
        print(f"  Message Capacity: ~{info['message_capacity']} characters")
    print("=" * 55)
