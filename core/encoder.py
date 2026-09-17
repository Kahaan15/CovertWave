import logging
import numpy as np
import random
from scipy.io import wavfile

logger = logging.getLogger(__name__)
from .audio_io import read_wav, require_integer_pcm
from .crypto import encrypt_message, derive_seed, get_encrypted_length


def get_random_positions(total_samples: int, num_positions: int, seed: int) -> list:
    """
    Generates unique random positions using password-derived seed.
    Same password = same positions every time (deterministic).
    """
    rng = random.Random(seed)
    positions = rng.sample(range(total_samples), num_positions)
    return positions


def encode_message(input_wav: str, output_wav: str, message: str, password: str, lsb_bits: int = 1):
    """
    Main encoding function.
    Hides encrypted message in audio file using randomized LSB technique.
    
    Args:
        input_wav  : Path to original WAV file
        output_wav : Path to save stego WAV file
        message    : Secret message to hide
        password   : Password for AES encryption + position seed
        lsb_bits   : Number of LSB bits to use (1, 2, or 4)
    """

    # --- Step 1: Read audio file ---
    sample_rate, samples = read_wav(input_wav)
    require_integer_pcm(samples, input_wav)

    # Handle stereo — flatten to mono view for embedding
    original_shape = samples.shape
    if samples.ndim == 2:
        flat_samples = samples[:, 0].copy()      # Use only left channel
    else:
        flat_samples = samples.copy()

    total_samples = len(flat_samples)

    # --- Step 2: Encrypt the message ---
    encrypted = encrypt_message(message, password)

    # --- Step 3: Prepend length header (4 bytes) ---
    # We store length of encrypted data so decoder knows when to stop
    length_header = len(encrypted).to_bytes(4, byteorder='big')
    payload = length_header + encrypted          # Total data to hide

    # --- Step 4: Convert payload to bits ---
    bits = []
    for byte in payload:
        for i in range(7, -1, -1):               # MSB to LSB
            bits.append((byte >> i) & 1)

    total_bits = len(bits)
    bits_per_sample = lsb_bits
    num_positions = (total_bits + bits_per_sample - 1) // bits_per_sample  # Ceiling division

    # --- Step 5: Capacity check ---
    if num_positions > total_samples:
        raise ValueError(
            f"Message too large! Need {num_positions} samples, "
            f"audio only has {total_samples} samples."
        )

    logger.info(f"[INFO] Message size     : {len(message)} characters")
    logger.info(f"[INFO] Encrypted size   : {len(encrypted)} bytes")
    logger.info(f"[INFO] Total bits       : {total_bits}")
    logger.info(f"[INFO] Samples used     : {num_positions} / {total_samples}")
    logger.info(f"[INFO] LSB bits used    : {lsb_bits}")

    # --- Step 6: Generate random positions using password seed ---
    seed = derive_seed(password)
    positions = get_random_positions(total_samples, num_positions, seed)

    # --- Step 7: Embed bits into audio samples ---
    flat_samples = flat_samples.astype(np.int32)  # Avoid overflow during bit ops
    bit_index = 0

    for pos in positions:
        if bit_index >= total_bits:
            break

        sample = int(flat_samples[pos])

        for b in range(lsb_bits):
            if bit_index >= total_bits:
                break
            bit = bits[bit_index]

            # Clear the b-th LSB and set it to our bit
            sample = (sample & ~(1 << b)) | (bit << b)
            bit_index += 1

        flat_samples[pos] = sample

    # --- Step 8: Rebuild audio array and save ---
    if samples.ndim == 2:
        samples[:, 0] = flat_samples.astype(samples.dtype)
    else:
        samples = flat_samples.astype(samples.dtype)

    wavfile.write(output_wav, sample_rate, samples)
    logger.info(f"[SUCCESS] Stego audio saved to: {output_wav}")


def calculate_capacity(input_wav: str, lsb_bits: int = 1) -> dict:
    """
    Calculates how much data can be hidden in the given WAV file.
    Returns capacity info as a dictionary.
    """
    sample_rate, samples = read_wav(input_wav)

    if samples.ndim == 2:
        total_samples = samples.shape[0]
    else:
        total_samples = len(samples)

    total_bits = total_samples * lsb_bits
    total_bytes = total_bits // 8
    usable_bytes = total_bytes - 4               # Subtract 4-byte length header
    usable_bytes = max(0, usable_bytes)

    # AES overhead: 16 bytes IV + padding, so actual message capacity is less
    # Rough estimate: subtract ~32 bytes for AES overhead
    message_capacity = max(0, usable_bytes - 32)

    return {
        "sample_rate"       : sample_rate,
        "total_samples"     : total_samples,
        "lsb_bits"          : lsb_bits,
        "total_bits"        : total_bits,
        "usable_bytes"      : usable_bytes,
        "message_capacity"  : message_capacity,
        "duration_seconds"  : total_samples / sample_rate
    }