import logging
import numpy as np
import random
from scipy.io import wavfile

logger = logging.getLogger(__name__)
from .audio_io import read_wav
from .crypto import decrypt_message, derive_seed


def get_random_positions(total_samples: int, num_positions: int, seed: int) -> list:
    """
    Regenerates the exact same random positions used during encoding.
    Same password = same seed = same positions (deterministic).
    """
    rng = random.Random(seed)
    positions = rng.sample(range(total_samples), num_positions)
    return positions


def decode_message(stego_wav: str, password: str, lsb_bits: int = 1) -> tuple:
    """
    Main decoding function.
    Extracts and decrypts hidden message from stego audio file.

    Args:
        stego_wav : Path to stego WAV file (with hidden message)
        password  : Same password used during encoding
        lsb_bits  : Same LSB bit count used during encoding (1, 2, or 4)

    Returns:
        (message, positions, total_samples) — the positions are returned so the
        UI can draw where the payload landed.
    """

    # --- Step 1: Read stego audio file ---
    sample_rate, samples = read_wav(stego_wav)

    # Handle stereo — use same channel as encoder (left channel)
    if samples.ndim == 2:
        flat_samples = samples[:, 0].copy()
    else:
        flat_samples = samples.copy()

    total_samples = len(flat_samples)
    flat_samples = flat_samples.astype(np.int32)

    # --- Step 2: First pass — extract length header (4 bytes = 32 bits) ---
    # We need to know how many bytes the encrypted payload is
    # Length header is always stored first, needs 32 bits / lsb_bits positions

    header_bits_needed = 32                      # 4 bytes * 8 bits
    header_positions_needed = (header_bits_needed + lsb_bits - 1) // lsb_bits

    seed = derive_seed(password)
    
    # Get enough positions to read the header
    rng = random.Random(seed)
    header_positions = rng.sample(range(total_samples), header_positions_needed)

    # Extract header bits
    header_bits = []
    for pos in header_positions:
        sample = int(flat_samples[pos])
        for b in range(lsb_bits):
            if len(header_bits) >= header_bits_needed:
                break
            header_bits.append((sample >> b) & 1)

    # Convert header bits to integer (payload length)
    payload_length = 0
    for bit in header_bits:
        payload_length = (payload_length << 1) | bit

    logger.info(f"[INFO] Detected payload length : {payload_length} bytes")

    # Sanity check
    if payload_length <= 0 or payload_length > total_samples:
        raise ValueError(
            "Invalid payload length detected. "
            "Wrong password or file is not a stego file."
        )

    # --- Step 3: Second pass — extract full payload ---
    total_bits_needed = (payload_length + 4) * 8  # +4 for header itself
    total_positions_needed = (total_bits_needed + lsb_bits - 1) // lsb_bits

    # Regenerate positions from scratch (same seed = same sequence)
    rng2 = random.Random(seed)
    all_positions = rng2.sample(range(total_samples), total_positions_needed)

    # Extract all bits
    all_bits = []
    for pos in all_positions:
        sample = int(flat_samples[pos])
        for b in range(lsb_bits):
            if len(all_bits) >= total_bits_needed:
                break
            all_bits.append((sample >> b) & 1)

    # --- Step 4: Convert bits back to bytes ---
    all_bytes = bytearray()
    for i in range(0, len(all_bits), 8):
        byte_bits = all_bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte_val = 0
        for bit in byte_bits:
            byte_val = (byte_val << 1) | bit
        all_bytes.append(byte_val)

    # --- Step 5: Separate header and encrypted payload ---
    encrypted_data = bytes(all_bytes[4:4 + payload_length])

    logger.info(f"[INFO] Extracted encrypted bytes : {len(encrypted_data)}")

    # --- Step 6: Decrypt using AES ---
    try:
        message = decrypt_message(encrypted_data, password)
        logger.info(f"[SUCCESS] Message decoded successfully!")
        return message, all_positions, total_samples
    except Exception as e:
        raise ValueError(
            "Decryption failed. Likely wrong password or corrupted file."
        ) from e