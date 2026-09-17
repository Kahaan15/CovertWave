"""
sequential.py — Naive sequential LSB embedding (the baseline).

This is the textbook method the paper's introduction describes: consecutive
samples modified in order, starting at sample 0, producing a contiguous block of
modified samples followed by an untouched remainder.

It is the arm the results section is supposed to compare against — the one whose
"clustered modification pattern creates exploitable statistical artifacts". No
implementation of it existed: research_benchmark.py called encode_message() (the
randomized encoder) for both the Sequential and the Randomized arm, so the two
columns in research_results_full.csv are the same algorithm run twice, differing
only by the random AES IV.
"""

import numpy as np

from .audio_io import read_wav, require_integer_pcm
from .payload import (
    HEADER_BYTES,
    assemble_payload,
    embed_bits,
    extract_bits,
    parse_payload,
    payload_to_bits,
    positions_needed,
    read_length_header,
)


def get_sequential_positions(total_samples: int, num_positions: int) -> list:
    """Positions 0, 1, 2, ... — no seed, no randomisation. That is the point."""
    if num_positions > total_samples:
        raise ValueError(
            f"Need {num_positions} positions but audio only has {total_samples} samples."
        )
    return list(range(num_positions))


def encode_sequential(
    input_wav: str,
    output_wav: str,
    message: str,
    password: str,
    lsb_bits: int = 1,
):
    """
    Embeds an AES-encrypted message into consecutive samples from the start of
    the file. Same payload format and same crypto as the other two methods, so
    the only variable is position selection.
    """
    sample_rate, samples = read_wav(input_wav)
    require_integer_pcm(samples, input_wav)

    flat = (samples[:, 0] if samples.ndim == 2 else samples).copy()
    total_samples = len(flat)

    payload = assemble_payload(message, password)
    bits    = payload_to_bits(payload)
    n_pos   = positions_needed(len(bits), lsb_bits)

    positions = get_sequential_positions(total_samples, n_pos)

    work = flat.astype(np.int32)
    embed_bits(work, positions, bits, lsb_bits)

    if samples.ndim == 2:
        samples[:, 0] = work.astype(samples.dtype)
    else:
        samples = work.astype(samples.dtype)

    from scipy.io import wavfile
    wavfile.write(output_wav, sample_rate, samples)


def decode_sequential(stego_wav: str, password: str, lsb_bits: int = 1):
    """
    Recovers a message embedded by encode_sequential().

    Returns (message, positions, total_samples) to match the other decoders.
    """
    _, samples = read_wav(stego_wav)
    flat = (samples[:, 0] if samples.ndim == 2 else samples).copy()
    total_samples = len(flat)
    work = flat.astype(np.int32)

    header_bits = extract_bits(
        work,
        get_sequential_positions(total_samples, positions_needed(32, lsb_bits)),
        32,
        lsb_bits,
    )
    payload_length = read_length_header(header_bits)

    if payload_length <= 0 or payload_length > total_samples:
        raise ValueError("Invalid payload length. Wrong password or not a stego file.")

    total_bits = (payload_length + HEADER_BYTES) * 8
    positions  = get_sequential_positions(
        total_samples, positions_needed(total_bits, lsb_bits)
    )
    all_bits = extract_bits(work, positions, total_bits, lsb_bits)

    try:
        message = parse_payload(all_bits, payload_length, password)
    except Exception as exc:
        raise ValueError("Decryption failed. Wrong password or corrupted file.") from exc

    return message, positions, total_samples
