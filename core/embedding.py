"""
embedding.py — Unified embedding engine parameterised by position strategy.

One code path. The strategy and the encryption flag are the only variables, so a
difference between benchmark arms cannot come from anywhere else.

This exists because of how the original study went wrong: research_benchmark.py
had a three-way if/elif in which the "Sequential" and "Randomized" branches both
called encode_message(), so two of the three reported arms were the same
algorithm and every sequential-vs-randomized conclusion in the paper described a
difference that was actually the random AES IV. Dispatching through a table of
strategies makes that class of mistake visible: the arms are data, not branches.

The 2x3 factorial {plaintext, AES} x {sequential, uniform, adaptive} is what
answers the reviewer request to disentangle the contribution of each component.
"""

import numpy as np
from scipy.io import wavfile

from .adaptive import get_adaptive_positions
from .audio_io import read_wav, require_integer_pcm
from .crypto import derive_seed
from .payload import (
    HEADER_BYTES,
    bits_to_bytes,
    embed_bits,
    extract_bits,
    payload_to_bits,
    positions_needed,
    read_length_header,
)
from .crypto import decrypt_message, encrypt_message

STRATEGIES = ("sequential", "uniform", "adaptive")


# -----------------------------------------------------------------------------
#  POSITION STRATEGIES
# -----------------------------------------------------------------------------

def _positions(strategy, samples, count, seed, lsb_bits, frame_size=512, min_weight=0.05):
    """Returns `count` embedding indices under the named strategy."""
    total = len(samples)
    if count > total:
        raise ValueError(
            f"Message too large: needs {count} samples, audio has {total}."
        )

    if strategy == "sequential":
        # Contiguous from the start of the file - the textbook naive method.
        return list(range(count))

    if strategy == "uniform":
        # Password-seeded uniform draw.
        #
        # CAUTION: random.sample is NOT prefix-stable in general. It switches
        # between a selection-set and a pool/partial-shuffle algorithm depending
        # on k relative to n, so sample(n, k1) is not always sample(n, k2)[:k1].
        # Concrete counterexample: n=300000, k1=500, k2=250000 disagree.
        #
        # The decoder relies on the prefix property (it draws the 32-bit header
        # first, then the payload). It holds here only because the header draw
        # is k=32, which takes the selection-set path and collides with a
        # previously drawn index with probability ~k^2/2n (< 1e-3 for these
        # covers). All 660 benchmark runs and 880 seed-variance runs extracted
        # exactly, but this is a probabilistic guarantee, not a structural one.
        # Enlarging the header, or shrinking the covers, raises the failure
        # odds. The adaptive branch below has no such caveat: the
        # Efraimidis-Spirakis race keys every sample independently of `count`,
        # so its prefix property is exact. Replacing this draw with the same
        # construction would fix it, at the cost of re-running the benchmark
        # (it would change every uniform-arm position set).
        import random
        return random.Random(seed).sample(range(total), count)

    if strategy == "adaptive":
        return get_adaptive_positions(
            samples, count, seed, frame_size, min_weight, lsb_bits
        )

    raise ValueError(f"Unknown position strategy: {strategy!r}")


# -----------------------------------------------------------------------------
#  PAYLOAD (with or without the crypto layer, for the ablation)
# -----------------------------------------------------------------------------

def _build_payload(message: str, password: str, encrypt: bool) -> bytes:
    if encrypt:
        body = encrypt_message(message, password)
    else:
        body = message.encode("utf-8")
    return len(body).to_bytes(HEADER_BYTES, byteorder="big") + body


def _read_payload(body: bytes, password: str, encrypt: bool) -> str:
    if encrypt:
        return decrypt_message(body, password)
    return body.decode("utf-8")


# -----------------------------------------------------------------------------
#  PUBLIC API
# -----------------------------------------------------------------------------

def embed(input_wav, output_wav, message, password, lsb_bits=1,
          strategy="adaptive", encrypt=True, frame_size=512, min_weight=0.05):
    """Embeds `message` into `input_wav` and writes the stego file."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    sample_rate, samples = read_wav(input_wav)
    require_integer_pcm(samples, input_wav)

    flat = (samples[:, 0] if samples.ndim == 2 else samples).copy()

    payload = _build_payload(message, password, encrypt)
    bits = payload_to_bits(payload)
    count = positions_needed(len(bits), lsb_bits)

    seed = derive_seed(password)
    positions = _positions(strategy, flat, count, seed, lsb_bits, frame_size, min_weight)

    work = flat.astype(np.int32)
    embed_bits(work, positions, bits, lsb_bits)

    if samples.ndim == 2:
        samples[:, 0] = work.astype(samples.dtype)
    else:
        samples = work.astype(samples.dtype)

    wavfile.write(output_wav, sample_rate, samples)
    return {"positions": count, "payload_bytes": len(payload), "bits": len(bits)}


def extract(stego_wav, password, lsb_bits=1, strategy="adaptive",
            encrypt=True, frame_size=512, min_weight=0.05):
    """Recovers a message embedded by embed() with the same parameters."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    _, samples = read_wav(stego_wav)
    flat = (samples[:, 0] if samples.ndim == 2 else samples).copy()
    total = len(flat)
    work = flat.astype(np.int32)
    seed = derive_seed(password)

    header_positions = _positions(
        strategy, flat, positions_needed(32, lsb_bits), seed, lsb_bits,
        frame_size, min_weight,
    )
    length = read_length_header(extract_bits(work, header_positions, 32, lsb_bits))

    if length <= 0 or length > total:
        raise ValueError("Invalid payload length. Wrong password or not a stego file.")

    total_bits = (length + HEADER_BYTES) * 8
    positions = _positions(
        strategy, flat, positions_needed(total_bits, lsb_bits), seed, lsb_bits,
        frame_size, min_weight,
    )
    all_bits = extract_bits(work, positions, total_bits, lsb_bits)
    body = bytes(bits_to_bytes(all_bits)[HEADER_BYTES:HEADER_BYTES + length])

    try:
        return _read_payload(body, password, encrypt)
    except Exception as exc:
        raise ValueError("Extraction failed. Wrong password or corrupted file.") from exc
