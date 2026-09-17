"""
test_covertwave.py — Regression tests.

The repository had no tests. That is the direct reason the two most serious
defects survived to publication:

  * the adaptive decoder failed on 30-55% of the corpus, returning corrupted
    plaintext without raising, and nothing ever asserted that a message came back;
  * the benchmark's "Sequential" and "Randomized" arms called the same function,
    and nothing ever asserted that two strategies produce different output.

Both of those are now assertions here.

Run:  python -m pytest tests/ -v
      python tests/test_covertwave.py        (no pytest needed)
"""

import os
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adaptive import encode_adaptive, decode_adaptive, get_adaptive_positions, lsb_cleared
from core.audio_io import read_mono, require_integer_pcm
from core.crypto import derive_seed
from core.embedding import STRATEGIES, embed, extract
from core.encoder import encode_message
from core.decoder import decode_message
from core.metrics import bit_error_rate, calculate_psnr
from core.payload import assemble_payload, bits_to_bytes, payload_to_bits
from core.sequential import encode_sequential, decode_sequential
from core.steganalysis import run_full_steganalysis, sample_pair_analysis

PASSWORD = "test_password"
MESSAGE = "The quick brown fox jumps over the lazy dog. 0123456789 éüñ"


# -----------------------------------------------------------------------------
#  Fixtures
# -----------------------------------------------------------------------------

def synthetic_cover(path, n=200_000, sample_rate=44100, stereo=False, seed=7):
    """A tonal signal plus noise — enough structure to behave like real audio."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    signal = (8000 * np.sin(2 * np.pi * 440 * t)
              + 3000 * np.sin(2 * np.pi * 1310 * t)
              + rng.normal(0, 300, n))
    samples = np.clip(signal, -32768, 32767).astype(np.int16)
    if stereo:
        samples = np.column_stack([samples, np.roll(samples, 17)])
    wavfile.write(path, sample_rate, samples)
    return path


def low_amplitude_cover(path, n=200_000, sample_rate=44100, seed=11):
    """
    A cover whose adjacent samples frequently land in the same value pair.

    Sample Pair Analysis derives its estimate from the pairs (u, v) that differ
    only in the LSB or not at all. A loud, fast-moving tone produces almost none
    of those (adjacent samples of a 440 Hz tone at amplitude 8000 differ by ~500),
    which makes the SPA quadratic degenerate and its output meaningless. Real
    audio has quiet passages and slow-moving low-frequency content, so W + Z sits
    around 3% of pairs. This fixture reproduces that condition.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    signal = 220 * np.sin(2 * np.pi * 55 * t) + rng.normal(0, 2.0, n)
    signal[: n // 4] *= 0.05                      # a near-silent passage
    samples = np.clip(signal, -32768, 32767).astype(np.int16)
    wavfile.write(path, sample_rate, samples)
    return path


def temp_wav(suffix=".wav"):
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.close()
    return handle.name


# -----------------------------------------------------------------------------
#  Payload plumbing
# -----------------------------------------------------------------------------

def test_bit_packing_roundtrips():
    payload = assemble_payload("hello", PASSWORD)
    assert bytes(bits_to_bytes(payload_to_bits(payload))) == payload


def test_bit_error_rate_bounds():
    assert bit_error_rate("abc", "abc") == 0.0
    assert bit_error_rate("abc", None) == 1.0
    assert 0.0 < bit_error_rate("abc", "abd") < 1.0
    # Truncation must count as error, not be silently ignored.
    assert bit_error_rate("abcdef", "abc") > 0.0


# -----------------------------------------------------------------------------
#  The adaptive decoder bug
# -----------------------------------------------------------------------------

def test_lsb_cleared_is_embedding_invariant():
    """
    The property the whole adaptive fix rests on: clearing the low bits makes
    cover and stego identical, so both sides derive the same weights.
    """
    cover = temp_wav()
    stego = temp_wav()
    try:
        synthetic_cover(cover)
        encode_adaptive(cover, stego, MESSAGE, PASSWORD, lsb_bits=1)

        a = lsb_cleared(read_mono(cover), 1)
        b = lsb_cleared(read_mono(stego), 1)
        assert np.array_equal(a, b), "LSB-cleared cover and stego must be identical"

        # ...and therefore the position draws must match exactly.
        seed = derive_seed(PASSWORD)
        pa = get_adaptive_positions(read_mono(cover), 500, seed, lsb_bits=1)
        pb = get_adaptive_positions(read_mono(stego), 500, seed, lsb_bits=1)
        assert pa == pb, "cover and stego must yield identical adaptive positions"
    finally:
        for path in (cover, stego):
            os.path.exists(path) and os.remove(path)


def test_adaptive_positions_are_prefix_stable():
    """
    The decoder draws 32 bits of header first and the full payload second. Those
    draws must agree on their overlap, or the header is read from the wrong
    samples.
    """
    cover = temp_wav()
    try:
        synthetic_cover(cover)
        samples = read_mono(cover)
        seed = derive_seed(PASSWORD)
        short = get_adaptive_positions(samples, 32, seed, lsb_bits=1)
        long = get_adaptive_positions(samples, 5000, seed, lsb_bits=1)
        assert short == long[:32], "position draw must be prefix-stable"
    finally:
        os.path.exists(cover) and os.remove(cover)


def test_adaptive_roundtrip_all_depths():
    cover = temp_wav()
    stego = temp_wav()
    try:
        synthetic_cover(cover)
        for lsb in (1, 2, 4):
            encode_adaptive(cover, stego, MESSAGE, PASSWORD, lsb_bits=lsb)
            recovered, _, _ = decode_adaptive(stego, PASSWORD, lsb_bits=lsb)
            assert recovered == MESSAGE, f"adaptive roundtrip failed at {lsb}-bit"
    finally:
        for path in (cover, stego):
            os.path.exists(path) and os.remove(path)


# -----------------------------------------------------------------------------
#  The identical-arms bug
# -----------------------------------------------------------------------------

def test_strategies_produce_different_stego():
    """
    The benchmark reported Sequential and Randomized as separate arms while
    calling the same function. Assert the three strategies actually differ.
    """
    cover = temp_wav()
    try:
        synthetic_cover(cover)
        outputs = {}
        for strategy in STRATEGIES:
            out = temp_wav()
            embed(cover, out, MESSAGE, PASSWORD, 1, strategy)
            outputs[strategy] = read_mono(out)
            os.remove(out)

        names = list(outputs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not np.array_equal(outputs[a], outputs[b]), \
                    f"strategies {a!r} and {b!r} produced identical stego audio"
    finally:
        os.path.exists(cover) and os.remove(cover)


def test_sequential_modifies_a_contiguous_prefix():
    """Sequential must actually be sequential — a contiguous run from sample 0."""
    cover = temp_wav()
    stego = temp_wav()
    try:
        synthetic_cover(cover)
        encode_sequential(cover, stego, MESSAGE, PASSWORD, lsb_bits=1)
        changed = np.flatnonzero(read_mono(cover) != read_mono(stego))
        assert len(changed) > 0
        assert changed[0] < 64, "sequential embedding must start near sample 0"
        span = changed[-1] - changed[0] + 1
        assert span < len(read_mono(cover)) * 0.5, \
            "sequential changes must be clustered, not spread over the file"
    finally:
        for path in (cover, stego):
            os.path.exists(path) and os.remove(path)


def test_unified_engine_roundtrips_every_arm():
    cover = temp_wav()
    stego = temp_wav()
    try:
        synthetic_cover(cover)
        for strategy in STRATEGIES:
            for encrypt in (True, False):
                for lsb in (1, 2, 4):
                    embed(cover, stego, MESSAGE, PASSWORD, lsb, strategy, encrypt)
                    got = extract(stego, PASSWORD, lsb, strategy, encrypt)
                    assert got == MESSAGE, \
                        f"{strategy}/encrypt={encrypt}/{lsb}-bit roundtrip failed"
    finally:
        for path in (cover, stego):
            os.path.exists(path) and os.remove(path)


# -----------------------------------------------------------------------------
#  Standard encoder, stereo, wrong password
# -----------------------------------------------------------------------------

def test_standard_roundtrip_and_stereo():
    for stereo in (False, True):
        cover = temp_wav()
        stego = temp_wav()
        try:
            synthetic_cover(cover, stereo=stereo)
            encode_message(cover, stego, MESSAGE, PASSWORD, lsb_bits=1)
            recovered, _, _ = decode_message(stego, PASSWORD, lsb_bits=1)
            assert recovered == MESSAGE, f"standard roundtrip failed (stereo={stereo})"

            if stereo:
                _, original = wavfile.read(cover)
                _, modified = wavfile.read(stego)
                assert np.array_equal(original[:, 1], modified[:, 1]), \
                    "right channel must be untouched"
        finally:
            for path in (cover, stego):
                os.path.exists(path) and os.remove(path)


def test_wrong_password_is_rejected():
    cover = temp_wav()
    stego = temp_wav()
    try:
        synthetic_cover(cover)
        embed(cover, stego, MESSAGE, PASSWORD, 1, "adaptive")
        try:
            got = extract(stego, "the_wrong_password", 1, "adaptive")
        except ValueError:
            return                      # expected
        assert got != MESSAGE, "wrong password must not recover the message"
    finally:
        for path in (cover, stego):
            os.path.exists(path) and os.remove(path)


def test_float_wav_is_rejected_not_silently_zeroed():
    """scipy returns float32 for IEEE-float WAVs; casting to int32 zeroes them."""
    path = temp_wav()
    try:
        wavfile.write(path, 44100, np.linspace(-1, 1, 1000).astype(np.float32))
        _, samples = wavfile.read(path)
        try:
            require_integer_pcm(samples, path)
        except ValueError:
            return                      # expected
        raise AssertionError("float WAV should have been rejected")
    finally:
        os.path.exists(path) and os.remove(path)


# -----------------------------------------------------------------------------
#  Metrics and steganalysis
# -----------------------------------------------------------------------------

def test_psnr_identical_files_is_infinite():
    cover = temp_wav()
    try:
        synthetic_cover(cover)
        assert calculate_psnr(cover, cover) == float("inf")
    finally:
        os.path.exists(cover) and os.remove(cover)


def test_spa_reads_near_zero_on_clean_audio():
    """
    The old estimator returned ~0.5 for every signal and so flagged all 44 clean
    covers as stego. The real one must sit near zero on a clean cover.
    """
    cover = temp_wav()
    try:
        low_amplitude_cover(cover)
        beta = sample_pair_analysis(samples=read_mono(cover))["beta"]
        assert abs(beta) < 0.02, f"SPA beta on clean audio should be ~0, got {beta}"
    finally:
        os.path.exists(cover) and os.remove(cover)


def test_spa_increases_with_payload():
    cover = temp_wav()
    stego = temp_wav()
    try:
        low_amplitude_cover(cover)
        clean = abs(sample_pair_analysis(samples=read_mono(cover))["beta"])
        capacity = len(read_mono(cover)) // 8
        embed(cover, stego, "X" * (capacity - 40), PASSWORD, 1, "uniform")
        loaded = abs(sample_pair_analysis(samples=read_mono(stego))["beta"])
        assert loaded > clean, (
            f"SPA must respond to embedding (clean {clean:.5f}, loaded {loaded:.5f})"
        )
    finally:
        for path in (cover, stego):
            os.path.exists(path) and os.remove(path)


def test_battery_reports_all_four_detectors():
    cover = temp_wav()
    try:
        synthetic_cover(cover)
        result = run_full_steganalysis(cover)
        for name in ("chi_square", "rs_analysis", "spa", "histogram"):
            assert name in result
            assert "detected" in result[name]
        summary = result["detection_summary"]
        assert 0 <= summary["methods_triggered"] <= 4
        assert summary["discriminative_triggered"] <= summary["discriminative_methods"]
    finally:
        os.path.exists(cover) and os.remove(cover)


# -----------------------------------------------------------------------------

def main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
