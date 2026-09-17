"""
payload.py — Shared payload assembly and bit-level LSB substitution.

The encoders differ ONLY in which sample indices they write to:

    sequential.py  positions 0, 1, 2, ... (contiguous)
    encoder.py     positions drawn uniformly at random from a password seed
    adaptive.py    positions drawn energy-weighted from the same seed

Everything else — AES, the length header, bit packing, the substitution itself —
is identical and lives here, so a difference between methods in the benchmark can
only come from the position strategy. The original code duplicated this logic in
each module, which is how the "Sequential" and "Randomized" benchmark arms ended
up calling the same function without anyone noticing.
"""

import numpy as np

from .crypto import encrypt_message, decrypt_message

HEADER_BYTES = 4          # big-endian length of the encrypted payload


def assemble_payload(message: str, password: str) -> bytes:
    """Builds D = L || IV || C (4-byte length header, 16-byte IV, ciphertext)."""
    encrypted = encrypt_message(message, password)
    return len(encrypted).to_bytes(HEADER_BYTES, byteorder="big") + encrypted


def payload_to_bits(payload: bytes) -> list:
    """Expands bytes to a flat MSB-first bit list."""
    bits = []
    for byte in payload:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits) -> bytearray:
    """Packs an MSB-first bit sequence back into bytes, dropping any partial tail."""
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        value = 0
        for bit in bits[i:i + 8]:
            value = (value << 1) | bit
        out.append(value)
    return out


def positions_needed(num_bits: int, lsb_bits: int) -> int:
    """Number of sample slots required to carry num_bits at lsb_bits per sample."""
    return (num_bits + lsb_bits - 1) // lsb_bits


def embed_bits(samples: np.ndarray, positions, bits, lsb_bits: int) -> None:
    """
    Writes `bits` into the low `lsb_bits` of `samples` at `positions`, in order.
    Mutates `samples` in place; it must be a signed integer array wide enough to
    hold the values (int32 for 16-bit PCM).
    """
    bit_index  = 0
    total_bits = len(bits)

    for pos in positions:
        if bit_index >= total_bits:
            break
        sample = int(samples[pos])
        for b in range(lsb_bits):
            if bit_index >= total_bits:
                break
            sample = (sample & ~(1 << b)) | (bits[bit_index] << b)
            bit_index += 1
        samples[pos] = sample


def extract_bits(samples: np.ndarray, positions, num_bits: int, lsb_bits: int) -> list:
    """Reads `num_bits` back out of `samples` at `positions`, in the same order."""
    bits = []
    for pos in positions:
        if len(bits) >= num_bits:
            break
        sample = int(samples[pos])
        for b in range(lsb_bits):
            if len(bits) >= num_bits:
                break
            bits.append((sample >> b) & 1)
    return bits


def parse_payload(all_bits, payload_length: int, password: str) -> str:
    """Strips the header, decrypts the ciphertext, returns the plaintext."""
    all_bytes = bits_to_bytes(all_bits)
    encrypted = bytes(all_bytes[HEADER_BYTES:HEADER_BYTES + payload_length])
    return decrypt_message(encrypted, password)


def read_length_header(header_bits) -> int:
    """Decodes the 32-bit big-endian length header."""
    length = 0
    for bit in header_bits:
        length = (length << 1) | bit
    return length
