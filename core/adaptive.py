"""
adaptive.py — Content-Adaptive Embedding Module
Gap 1 Implementation: Energy-aware position selection for LSB steganography.

Core idea:
- Human ear is less sensitive to distortion in loud/high-energy audio regions.
- Silent or quiet regions are very sensitive — any change is noticeable.
- Instead of embedding uniformly at random positions, we WEIGHT positions
  by their local energy so that loud frames get more bits, quiet frames get fewer.

This is the novel contribution of the paper:
  "Adaptive, energy-weighted randomized LSB position selection"
  — combines two techniques (randomization + energy-awareness) not previously
    combined in published work.
"""

import numpy as np
import random
from scipy.io import wavfile


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME ENERGY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_frame_energy(samples: np.ndarray, frame_size: int = 512) -> np.ndarray:
    """
    Splits audio into frames and computes RMS energy per frame.

    Args:
        samples    : 1D array of audio samples (int or float)
        frame_size : Number of samples per frame (default 512 ≈ 11ms at 44.1kHz)

    Returns:
        energy : 1D float array, one energy value per frame
    """
    samples = samples.astype(np.float64)
    num_frames = len(samples) // frame_size

    energy = np.zeros(num_frames, dtype=np.float64)

    for i in range(num_frames):
        frame = samples[i * frame_size : (i + 1) * frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        energy[i] = rms

    return energy


def energy_to_sample_weights(
    samples: np.ndarray,
    frame_size: int = 512,
    min_weight: float = 0.05
) -> np.ndarray:
    """
    Maps per-frame energy values to per-sample embedding weights.

    Logic:
        - High energy sample → high weight → more likely to be selected for embedding
        - Low energy / silent sample → low weight → rarely selected
        - min_weight ensures even silent regions have a small chance (fallback)

    Args:
        samples    : 1D audio sample array
        frame_size : Frame size used for energy computation
        min_weight : Minimum weight for any sample (prevents zero probability)

    Returns:
        weights : 1D float array, same length as samples, normalized to sum to 1.0
    """
    energy = compute_frame_energy(samples, frame_size)
    num_frames = len(energy)

    # Assign each sample its frame's energy value
    weights = np.zeros(len(samples), dtype=np.float64)
    for i in range(num_frames):
        start = i * frame_size
        end   = min(start + frame_size, len(samples))
        weights[start:end] = energy[i]

    # Handle trailing samples not covered by a full frame
    if len(samples) > num_frames * frame_size:
        trailing_start = num_frames * frame_size
        trailing_frame = samples[trailing_start:].astype(np.float64)
        trailing_rms   = np.sqrt(np.mean(trailing_frame ** 2)) if len(trailing_frame) > 0 else 0.0
        weights[trailing_start:] = trailing_rms

    # Enforce minimum weight (avoid zero-probability silent regions)
    weights = np.maximum(weights, min_weight * np.max(weights) if np.max(weights) > 0 else min_weight)

    # Normalize to probability distribution (sum = 1.0)
    total = np.sum(weights)
    if total > 0:
        weights = weights / total
    else:
        # Fallback: uniform distribution
        weights = np.ones(len(samples)) / len(samples)

    return weights


# ─────────────────────────────────────────────────────────────────────────────
#  ADAPTIVE POSITION SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_adaptive_positions(
    samples: np.ndarray,
    num_positions: int,
    seed: int,
    frame_size: int = 512,
    min_weight: float = 0.05
) -> list:
    """
    Selects embedding positions weighted by local audio energy.

    High-energy (loud) samples are preferred for embedding.
    Low-energy (quiet/silent) samples are rarely selected.

    This is the KEY difference from uniform randomized LSB:
        - Uniform randomized: positions drawn with equal probability
        - Adaptive randomized: positions drawn with energy-weighted probability

    Args:
        samples       : 1D audio sample array
        num_positions : Number of positions needed
        seed          : Deterministic seed (derived from password)
        frame_size    : Frame size for energy analysis
        min_weight    : Minimum probability weight for silent samples

    Returns:
        positions : Sorted list of unique sample indices
    """
    total_samples = len(samples)

    if num_positions > total_samples:
        raise ValueError(
            f"Need {num_positions} positions but audio only has {total_samples} samples."
        )

    # Compute energy-based weights
    weights = energy_to_sample_weights(samples, frame_size, min_weight)

    # Use seeded numpy RNG for deterministic weighted sampling
    rng = np.random.default_rng(seed)

    # Weighted sampling WITHOUT replacement
    positions = rng.choice(
        total_samples,
        size=num_positions,
        replace=False,
        p=weights
    )

    return positions.tolist()


def get_uniform_positions(
    total_samples: int,
    num_positions: int,
    seed: int
) -> list:
    """
    Standard uniform random position selection (original method).
    Kept here for direct comparison with adaptive method.
    """
    rng = random.Random(seed)
    return rng.sample(range(total_samples), num_positions)


# ─────────────────────────────────────────────────────────────────────────────
#  ADAPTIVE ENCODE (main function called by encoder.py)
# ─────────────────────────────────────────────────────────────────────────────

def encode_adaptive(
    input_wav: str,
    output_wav: str,
    message: str,
    password: str,
    lsb_bits: int = 1,
    frame_size: int = 512,
    min_weight: float = 0.05
):
    """
    Full adaptive encoding pipeline.
    Drop-in replacement for encode_message() in encoder.py.

    Args:
        input_wav  : Original WAV file path
        output_wav : Output stego WAV file path
        message    : Secret message string
        password   : Encryption + position-seed password
        lsb_bits   : 1, 2, or 4 LSB bits per sample
        frame_size : Frame size for energy analysis (default 512)
        min_weight : Minimum embedding probability for silent regions
    """
    from scipy.io import wavfile as wf
    from .crypto import encrypt_message, derive_seed

    # --- Read audio ---
    sample_rate, samples = wf.read(input_wav)
    original_shape = samples.shape

    if samples.ndim == 2:
        flat_samples = samples[:, 0].copy()
    else:
        flat_samples = samples.copy()

    total_samples = len(flat_samples)

    # --- Encrypt message ---
    encrypted  = encrypt_message(message, password)
    length_hdr = len(encrypted).to_bytes(4, byteorder='big')
    payload    = length_hdr + encrypted

    # --- Convert to bits ---
    bits = []
    for byte in payload:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    total_bits   = len(bits)
    num_positions = (total_bits + lsb_bits - 1) // lsb_bits

    if num_positions > total_samples:
        raise ValueError(
            f"Message too large! Need {num_positions} samples, "
            f"audio only has {total_samples}."
        )

    print(f"[ADAPTIVE] Message     : {len(message)} chars")
    print(f"[ADAPTIVE] Encrypted   : {len(encrypted)} bytes")
    print(f"[ADAPTIVE] Total bits  : {total_bits}")
    print(f"[ADAPTIVE] Samples used: {num_positions} / {total_samples}")
    print(f"[ADAPTIVE] Frame size  : {frame_size}")
    print(f"[ADAPTIVE] Min weight  : {min_weight}")

    # --- Energy analysis ---
    seed      = derive_seed(password)
    work_arr  = flat_samples.astype(np.float64)
    positions = get_adaptive_positions(work_arr, num_positions, seed, frame_size, min_weight)

    # --- Embed bits ---
    flat_samples = flat_samples.astype(np.int32)
    bit_index    = 0

    for pos in positions:
        if bit_index >= total_bits:
            break
        sample = int(flat_samples[pos])
        for b in range(lsb_bits):
            if bit_index >= total_bits:
                break
            bit    = bits[bit_index]
            sample = (sample & ~(1 << b)) | (bit << b)
            bit_index += 1
        flat_samples[pos] = sample

    # --- Save stego file ---
    if samples.ndim == 2:
        samples[:, 0] = flat_samples.astype(samples.dtype)
    else:
        samples = flat_samples.astype(samples.dtype)

    wf.write(output_wav, sample_rate, samples)
    print(f"[ADAPTIVE] Stego saved : {output_wav}")


def decode_adaptive(
    stego_wav: str,
    password: str,
    lsb_bits: int = 1,
    frame_size: int = 512,
    min_weight: float = 0.05
) -> str:
    """
    Decodes a message hidden using encode_adaptive().
    Must use identical frame_size and min_weight as encoding.

    Args:
        stego_wav  : Path to stego WAV file
        password   : Same password used during encoding
        lsb_bits   : Same LSB bit count used during encoding
        frame_size : Same frame size used during encoding
        min_weight : Same min_weight used during encoding

    Returns:
        Decrypted secret message string
    """
    from scipy.io import wavfile as wf
    from .crypto import decrypt_message, derive_seed

    # --- Read stego audio ---
    sample_rate, samples = wf.read(stego_wav)

    if samples.ndim == 2:
        flat_samples = samples[:, 0].copy()
    else:
        flat_samples = samples.copy()

    total_samples = len(flat_samples)
    flat_samples  = flat_samples.astype(np.int32)

    seed = derive_seed(password)

    # --- Pass 1: extract 32-bit length header ---
    header_bits_needed     = 32
    header_positions_needed = (header_bits_needed + lsb_bits - 1) // lsb_bits

    work_arr = flat_samples.astype(np.float64)
    all_header_positions = get_adaptive_positions(
        work_arr, header_positions_needed, seed, frame_size, min_weight
    )

    header_bits = []
    for pos in all_header_positions:
        sample = int(flat_samples[pos])
        for b in range(lsb_bits):
            if len(header_bits) >= header_bits_needed:
                break
            header_bits.append((sample >> b) & 1)

    payload_length = 0
    for bit in header_bits:
        payload_length = (payload_length << 1) | bit

    print(f"[ADAPTIVE] Detected payload length: {payload_length} bytes")

    if payload_length <= 0 or payload_length > total_samples:
        raise ValueError("Invalid payload length. Wrong password or not a stego file.")

    # --- Pass 2: extract full payload ---
    total_bits_needed     = (payload_length + 4) * 8
    total_positions_needed = (total_bits_needed + lsb_bits - 1) // lsb_bits

    all_positions = get_adaptive_positions(
        work_arr, total_positions_needed, seed, frame_size, min_weight
    )

    all_bits = []
    for pos in all_positions:
        sample = int(flat_samples[pos])
        for b in range(lsb_bits):
            if len(all_bits) >= total_bits_needed:
                break
            all_bits.append((sample >> b) & 1)

    # --- Convert bits to bytes ---
    all_bytes = bytearray()
    for i in range(0, len(all_bits), 8):
        byte_bits = all_bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte_val = 0
        for bit in byte_bits:
            byte_val = (byte_val << 1) | bit
        all_bytes.append(byte_val)

    encrypted_data = bytes(all_bytes[4:4 + payload_length])
    print(f"[ADAPTIVE] Extracted encrypted bytes: {len(encrypted_data)}")

    try:
        message = decrypt_message(encrypted_data, password)
        print(f"[ADAPTIVE] Decode successful!")
        return message, all_positions, total_samples
    except Exception as e:
        raise ValueError("Decryption failed. Wrong password or corrupted file.") from e


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS UTILITIES (for paper metrics)
# ─────────────────────────────────────────────────────────────────────────────

def get_energy_profile(wav_path: str, frame_size: int = 512) -> dict:
    """
    Returns energy profile of a WAV file for analysis and visualization.
    Used to generate per-frame SNR plots in the paper.

    Returns dict with:
        energy     : per-frame RMS energy array
        frame_times: timestamps for each frame (seconds)
        sample_rate: audio sample rate
        stats      : summary statistics
    """
    sample_rate, samples = wavfile.read(wav_path)

    if samples.ndim == 2:
        samples = samples[:, 0]

    energy      = compute_frame_energy(samples, frame_size)
    frame_times = np.arange(len(energy)) * frame_size / sample_rate

    return {
        "energy"      : energy,
        "frame_times" : frame_times,
        "sample_rate" : sample_rate,
        "frame_size"  : frame_size,
        "stats": {
            "max_energy"  : float(np.max(energy)),
            "min_energy"  : float(np.min(energy)),
            "mean_energy" : float(np.mean(energy)),
            "std_energy"  : float(np.std(energy)),
            "silent_frames": int(np.sum(energy < 0.01 * np.max(energy))),
            "total_frames" : len(energy),
        }
    }


def compare_embedding_distributions(
    wav_path: str,
    num_positions: int,
    seed: int,
    frame_size: int = 512
) -> dict:
    """
    Compares where uniform vs. adaptive methods place their embedding positions.
    Used to generate the position distribution figure in the paper.

    Returns:
        uniform_energies  : energy values at uniformly selected positions
        adaptive_energies : energy values at adaptively selected positions
        (adaptive should be systematically higher — proof of the algorithm working)
    """
    sample_rate, samples = wavfile.read(wav_path)

    if samples.ndim == 2:
        samples = samples[:, 0]

    samples_f = samples.astype(np.float64)

    # Get positions from both methods
    uniform_pos  = get_uniform_positions(len(samples), num_positions, seed)
    adaptive_pos = get_adaptive_positions(samples_f, num_positions, seed, frame_size)

    # Get energy values at each set of positions
    weights = energy_to_sample_weights(samples_f, frame_size)
    # Unnormalized energy per sample
    energy_per_sample = weights * len(samples)  # reverse normalize for readability

    uniform_energies  = energy_per_sample[uniform_pos]
    adaptive_energies = energy_per_sample[adaptive_pos]

    return {
        "uniform_mean_energy"  : float(np.mean(uniform_energies)),
        "adaptive_mean_energy" : float(np.mean(adaptive_energies)),
        "uniform_energies"     : uniform_energies,
        "adaptive_energies"    : adaptive_energies,
        "improvement_ratio"    : float(np.mean(adaptive_energies) / np.mean(uniform_energies))
                                  if np.mean(uniform_energies) > 0 else 1.0
    }