import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from encoder import encode_message
from decoder import decode_message
import os
import tempfile


def calculate_psnr(original_wav: str, stego_wav: str) -> float:
    """
    Calculates PSNR (Peak Signal-to-Noise Ratio) between original
    and stego audio files.

    Higher PSNR = better quality = less distortion.
    PSNR > 40 dB is generally considered imperceptible to human ear.

    Formula: PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    """

    # --- Read both files ---
    _, original = wavfile.read(original_wav)
    _, stego    = wavfile.read(stego_wav)

    # --- Use left channel if stereo ---
    if original.ndim == 2:
        original = original[:, 0]
    if stego.ndim == 2:
        stego = stego[:, 0]

    # --- Convert to float for calculation ---
    original = original.astype(np.float64)
    stego    = stego.astype(np.float64)

    # --- Calculate MSE (Mean Squared Error) ---
    mse = np.mean((original - stego) ** 2)

    if mse == 0:
        return float('inf')                      # Files are identical

    # --- MAX value depends on bit depth ---
    # 16-bit audio: max = 32767
    max_val = 32767.0

    # --- Calculate PSNR ---
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return round(psnr, 4)


def calculate_mse(original_wav: str, stego_wav: str) -> float:
    """
    Calculates Mean Squared Error between original and stego audio.
    Lower MSE = less distortion.
    """
    _, original = wavfile.read(original_wav)
    _, stego    = wavfile.read(stego_wav)

    if original.ndim == 2:
        original = original[:, 0]
    if stego.ndim == 2:
        stego = stego[:, 0]

    original = original.astype(np.float64)
    stego    = stego.astype(np.float64)

    mse = np.mean((original - stego) ** 2)
    return round(mse, 6)


def run_psnr_experiment(input_wav: str, message: str, password: str) -> dict:
    """
    Runs PSNR experiment across all 3 LSB modes (1-bit, 2-bit, 4-bit).
    Creates temporary stego files, measures PSNR and MSE for each mode.

    Returns dictionary with results for all modes.
    """
    results = {}
    lsb_modes = [1, 2, 4]

    print("\n[EXPERIMENT] Running PSNR across LSB modes...")
    print("-" * 50)

    for lsb in lsb_modes:
        # Create a temporary stego file
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False
        )
        temp_path = temp_file.name
        temp_file.close()

        try:
            # Encode with current LSB mode
            encode_message(input_wav, temp_path, message, password, lsb_bits=lsb)

            # Calculate metrics
            psnr = calculate_psnr(input_wav, temp_path)
            mse  = calculate_mse(input_wav, temp_path)

            results[lsb] = {
                "lsb_bits" : lsb,
                "psnr"     : psnr,
                "mse"      : mse
            }

            print(f"[LSB={lsb}] PSNR: {psnr} dB | MSE: {mse}")

        finally:
            # Always clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    print("-" * 50)
    return results


def plot_psnr_graph(results: dict, save_path: str = "psnr_graph.png"):
    """
    Plots PSNR vs LSB mode bar chart and saves as PNG.
    This graph goes directly into your research paper.
    """
    lsb_modes = list(results.keys())
    psnr_vals = [results[lsb]["psnr"] for lsb in lsb_modes]
    mse_vals  = [results[lsb]["mse"]  for lsb in lsb_modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- PSNR Bar Chart ---
    colors_psnr = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    bars = ax1.bar(
        [f"{lsb}-bit LSB" for lsb in lsb_modes],
        psnr_vals,
        color=colors_psnr,
        edgecolor='black',
        width=0.5
    )
    ax1.set_title("PSNR vs LSB Mode", fontsize=14, fontweight='bold')
    ax1.set_xlabel("LSB Mode", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.set_ylim(0, max(psnr_vals) + 10)

    # Add value labels on bars
    for bar, val in zip(bars, psnr_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val} dB",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    # Add reference line at 40 dB (imperceptibility threshold)
    ax1.axhline(
        y=40, color='blue', linestyle='--',
        linewidth=1.5, label='Imperceptibility threshold (40 dB)'
    )
    ax1.legend(fontsize=9)

    # --- MSE Bar Chart ---
    colors_mse = ['#3498db', '#9b59b6', '#e67e22']
    bars2 = ax2.bar(
        [f"{lsb}-bit LSB" for lsb in lsb_modes],
        mse_vals,
        color=colors_mse,
        edgecolor='black',
        width=0.5
    )
    ax2.set_title("MSE vs LSB Mode", fontsize=14, fontweight='bold')
    ax2.set_xlabel("LSB Mode", fontsize=12)
    ax2.set_ylabel("Mean Squared Error", fontsize=12)

    # Add value labels on bars
    for bar, val in zip(bars2, mse_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val}",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    plt.suptitle(
        "Audio Quality Analysis — Steganography LSB Modes",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n[INFO] Graph saved to: {save_path}")


def print_capacity_report(input_wav: str):
    """
    Prints a full capacity report for the given WAV file
    across all 3 LSB modes.
    """
    from encoder import calculate_capacity

    print("\n[CAPACITY REPORT]")
    print("=" * 55)

    for lsb in [1, 2, 4]:
        info = calculate_capacity(input_wav, lsb_bits=lsb)
        print(f"\n  LSB Mode        : {lsb}-bit")
        print(f"  Duration        : {info['duration_seconds']:.2f} seconds")
        print(f"  Total Samples   : {info['total_samples']}")
        print(f"  Usable Bytes    : {info['usable_bytes']}")
        print(f"  Message Capacity: ~{info['message_capacity']} characters")

    print("=" * 55)