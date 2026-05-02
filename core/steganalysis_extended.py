import numpy as np
from scipy.io import wavfile
from scipy.stats import chisquare
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def chi_square_test(wav_path: str, chunk_size: int = 1024) -> dict:
    """
    Chi-Square Statistical Test (PoVs Analysis).
    Detects if LSB frequencies are unnaturally equalized.
    """
    samples = _read_samples(wav_path)
    lsbs = samples & 1
    
    # Observe pairs of values (2k, 2k+1)
    # Simplified version: check if LSBs are too close to a 50/50 distribution in chunks
    total = len(lsbs)
    num_chunks = total // chunk_size
    
    p_values = []
    for i in range(num_chunks):
        chunk = lsbs[i*chunk_size : (i+1)*chunk_size]
        obs = np.bincount(chunk, minlength=2)
        exp = [chunk_size / 2, chunk_size / 2]
        _, p = chisquare(obs, f_exp=exp)
        p_values.append(p)
    
    avg_p = np.mean(p_values)
    # For stego, p-value tends towards 1.0 (very equal distribution)
    detected = avg_p > 0.9 
    verdict = "STEGO DETECTED" if detected else "CLEAN"
    
    return {
        "method": "Chi-Square Test",
        "avg_p_value": round(float(avg_p), 4),
        "verdict": verdict,
        "detected": detected
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: READ AUDIO
# ─────────────────────────────────────────────────────────────────────────────

def _read_samples(wav_path: str) -> np.ndarray:
    """Reads WAV robustly, handles inconsistent headers, returns left channel."""
    try:
        # Try standard scipy read
        _, samples = wavfile.read(wav_path)
    except ValueError:
        # Fallback for inconsistent headers (nAvgBytesPerSec issues)
        # We use numpy to read raw data if scipy fails, or just wrap in a more lenient way
        import soundfile as sf
        samples, _ = sf.read(wav_path, dtype='int16')
    except ImportError:
        # If soundfile isn't there, we'll try to at least skip the file gracefully
        raise

    if samples.ndim == 2:
        samples = samples[:, 0]
    return samples.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 1: RS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _discriminant(group: np.ndarray) -> float:
    """
    Smoothness discriminant function f(x).
    Measures how smooth a group of samples is.
    f = sum of |x[i+1] - x[i]| for all adjacent pairs.
    Lower value = smoother = more natural audio.
    """
    return float(np.sum(np.abs(np.diff(group.astype(np.float64))))  )


def _flip_lsb(group: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Applies LSB flip to positions where mask == 1."""
    flipped = group.copy()
    flipped[mask == 1] = flipped[mask == 1] ^ 1   # XOR with 1 flips LSB
    return flipped


def _shift_lsb(group: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies -1 shift (inverse flip) to positions where mask == 1.
    Used for the negative mask in RS analysis.
    Maps: even → odd (n → n-1), odd → even (n → n+1), i.e. toggles LSB differently.
    """
    shifted = group.copy().astype(np.int32)
    for i, m in enumerate(mask):
        if m == 1:
            val = shifted[i]
            # F_{-1}: even values go down by 1, odd values go up by 1
            if val % 2 == 0:
                shifted[i] = val - 1
            else:
                shifted[i] = val + 1
    return shifted.astype(np.int32)


def rs_analysis(wav_path: str, group_size: int = 4) -> dict:
    """
    Vectorized RS (Regular-Singular) Analysis.
    Optimized for speed using NumPy slicing.
    """
    samples = _read_samples(wav_path)
    n = len(samples)
    num_groups = n // group_size
    
    # Reshape into groups: (num_groups, group_size)
    groups = samples[:num_groups * group_size].reshape((num_groups, group_size)).astype(np.float64)
    
    def get_rs_stats(work_groups, is_negative=False):
        # Discriminant: sum(|x[i+1] - x[i]|)
        d_orig = np.sum(np.abs(np.diff(work_groups, axis=1)), axis=1)
        
        # Apply mask [0, 1, 0, 1]
        mask = np.tile([0, 1], group_size // 2)[:group_size]
        
        flipped = work_groups.copy()
        if not is_negative:
            # F1 flip: x XOR 1
            flipped[:, mask == 1] = flipped[:, mask == 1].astype(np.int32) ^ 1
        else:
            # F-1 shift: even -> n-1, odd -> n+1
            m_ones = (mask == 1)
            vals = flipped[:, m_ones].astype(np.int32)
            evens = (vals % 2 == 0)
            vals[evens] -= 1
            vals[~evens] += 1
            flipped[:, m_ones] = vals
            
        d_flipped = np.sum(np.abs(np.diff(flipped.astype(np.float64), axis=1)), axis=1)
        
        R = np.sum(d_flipped > d_orig)
        S = np.sum(d_flipped < d_orig)
        U = np.sum(d_flipped == d_orig)
        return R, S, U

    Rm, Sm, Um = get_rs_stats(groups, is_negative=False)
    Rn, Sn, Un = get_rs_stats(groups, is_negative=True)

    rm, sm = Rm / num_groups, Sm / num_groups
    rn, sn = Rn / num_groups, Sn / num_groups

    d0, d1 = rm - rn, sm - sn
    estimated_rate = abs(d0 / (d0 - d1)) if abs(d0 - d1) > 1e-10 else 0.0
    estimated_rate = min(max(estimated_rate, 0.0), 1.0)

    rs_diff = abs((rm - sm) - (rn - sn))
    detected = rs_diff >= 0.08
    verdict = "CLEAN" if rs_diff < 0.02 else ("SUSPICIOUS" if rs_diff < 0.08 else "STEGO DETECTED")

    return {
        "method": "RS Analysis",
        "Rm": Rm, "Sm": Sm, "Um": Um,
        "Rn": Rn, "Sn": Sn, "Un": Un,
        "rm": round(rm, 4), "sm": round(sm, 4),
        "rn": round(rn, 4), "sn": round(sn, 4),
        "rs_diff": round(rs_diff, 6),
        "estimated_rate": round(estimated_rate, 4),
        "verdict": verdict,
        "detected": detected
    }


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 2: SAMPLE PAIR ANALYSIS (SPA)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_mse(original_wav: str, stego_wav: str) -> float:
    """Calculates MSE robustly."""
    original = _read_samples_robust(original_wav).astype(np.float64)
    stego = _read_samples_robust(stego_wav).astype(np.float64)
    return float(np.mean((original[:len(stego)] - stego)**2))

def sample_pair_analysis(wav_path: str) -> dict:
    """
    Sample Pair Analysis (SPA).

    Theory:
        Examines adjacent sample pairs (u, v).
        Counts pairs where:
            W  = count of pairs where u is even and v is odd (or vice versa) — "cross pairs"
            Z  = count of pairs where both are even or both are odd — "same parity pairs"

        For clean audio: W and Z have a specific natural ratio.
        After LSB embedding: this ratio shifts predictably.
        The shift can be used to estimate the fraction of samples modified (beta).

        If beta > 0.05 (5% of samples changed) → likely stego

    Args:
        wav_path : Path to WAV file

    Returns:
        dict with pair counts, estimated beta (modification rate), verdict
    """
    samples = _read_samples(wav_path)
    n       = len(samples) - 1   # number of adjacent pairs

    # Count pair types
    u = samples[:-1].astype(np.int64)
    v = samples[1:].astype(np.int64)

    u_lsb = u & 1
    v_lsb = v & 1

    # W: cross-parity pairs (one even, one odd)
    W = int(np.sum(u_lsb != v_lsb))
    # Z: same-parity pairs (both even or both odd)
    Z = int(np.sum(u_lsb == v_lsb))

    # Estimate beta (fraction of samples that carry hidden bits)
    # From Dumitrescu et al. SPA formula:
    # beta ≈ (W - Z) / (2 * n) + 0.5   (simplified linear form)
    if n > 0:
        beta = (W - Z) / (2 * n) + 0.5
        beta = min(max(beta, 0.0), 1.0)
    else:
        beta = 0.0

    # Estimated payload size (fraction of total samples carrying hidden data)
    estimated_payload_fraction = beta

    # Detection verdict
    if beta < 0.05:
        verdict  = "CLEAN — modification rate below threshold"
        detected = False
    elif beta < 0.15:
        verdict  = "SUSPICIOUS — low embedding rate detected"
        detected = False
    else:
        verdict  = "STEGO DETECTED — significant sample modification"
        detected = True

    return {
        "method"                    : "Sample Pair Analysis",
        "total_pairs"               : n,
        "cross_parity_pairs_W"      : W,
        "same_parity_pairs_Z"       : Z,
        "W_ratio"                   : round(W / n, 4) if n > 0 else 0,
        "Z_ratio"                   : round(Z / n, 4) if n > 0 else 0,
        "estimated_beta"            : round(beta, 4),
        "estimated_payload_fraction": round(estimated_payload_fraction, 4),
        "verdict"                   : verdict,
        "detected"                  : detected
    }


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 3: HISTOGRAM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def histogram_analysis(wav_path: str) -> dict:
    """
    Histogram / LSB-pair Analysis.

    Theory:
        In clean audio, the histogram of sample values follows the audio's
        natural amplitude distribution (usually bell-shaped / Laplacian).

        Sequential LSB embedding creates a very specific artifact:
            Values 2k and 2k+1 (each even-odd pair) get equalized — their
            counts become nearly equal because embedding randomly flips LSBs.
            This is called the "pairs effect" or histogram equalization artifact.

        Randomized LSB embedding partially breaks this pattern (that's our point),
        but high payload rates still leave a detectable signature.

    Metric:
        For each adjacent value pair (2k, 2k+1), compute |count(2k) - count(2k+1)|.
        Clean audio: large differences (natural distribution)
        Stego audio: small differences (pairs equalized by embedding)

    Args:
        wav_path : Path to WAV file

    Returns:
        dict with histogram statistics, pair difference metrics, verdict
    """
    samples = _read_samples(wav_path)

    # Build value histogram (16-bit audio: values -32768 to 32767)
    sample_min = int(np.min(samples))
    sample_max = int(np.max(samples))

    # Use full range histogram
    hist, bin_edges = np.histogram(samples, bins=256, range=(sample_min, sample_max))

    # Pair analysis on histogram bins
    # Compare adjacent bin pairs
    pair_diffs = []
    for i in range(0, len(hist) - 1, 2):
        diff = abs(int(hist[i]) - int(hist[i + 1]))
        pair_diffs.append(diff)

    pair_diffs      = np.array(pair_diffs, dtype=np.float64)
    mean_pair_diff  = float(np.mean(pair_diffs))
    std_pair_diff   = float(np.std(pair_diffs))

    # Normalization: relative to total samples
    normalized_diff = mean_pair_diff / len(samples) if len(samples) > 0 else 0.0

    # LSB histogram: direct count of 0s and 1s in LSB layer
    lsbs        = samples & 1
    lsb_zeros   = int(np.sum(lsbs == 0))
    lsb_ones    = int(np.sum(lsbs == 1))
    lsb_balance = abs(lsb_zeros - lsb_ones) / len(samples)

    # Detection: low mean pair diff + low LSB balance = suspicious
    if normalized_diff < 0.001 and lsb_balance < 0.02:
        verdict  = "STEGO DETECTED — histogram pairs equalized"
        detected = True
    elif normalized_diff < 0.005 or lsb_balance < 0.05:
        verdict  = "SUSPICIOUS — partial histogram anomaly"
        detected = False
    else:
        verdict  = "CLEAN — histogram distribution natural"
        detected = False

    return {
        "method"          : "Histogram Analysis",
        "total_samples"   : len(samples),
        "mean_pair_diff"  : round(mean_pair_diff, 4),
        "std_pair_diff"   : round(std_pair_diff, 4),
        "normalized_diff" : round(normalized_diff, 6),
        "lsb_zeros"       : lsb_zeros,
        "lsb_ones"        : lsb_ones,
        "lsb_balance"     : round(lsb_balance, 6),
        "verdict"         : verdict,
        "detected"        : detected,
        "hist"            : hist,
        "bin_edges"       : bin_edges
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FULL BATTERY: RUN ALL 4 METHODS
# ─────────────────────────────────────────────────────────────────────────────

def _read_samples_robust(wav_path: str) -> np.ndarray:
    """Helper for metrics to read samples even with bad headers."""
    try:
        _, samples = wavfile.read(wav_path)
    except:
        import soundfile as sf
        samples, _ = sf.read(wav_path, dtype='int16')
    
    if samples.ndim == 2:
        samples = samples[:, 0]
    return samples.astype(np.int32)

def calculate_psnr(original_wav: str, stego_wav: str) -> float:
    """Calculates PSNR robustly."""
    original = _read_samples_robust(original_wav).astype(np.float64)
    stego = _read_samples_robust(stego_wav).astype(np.float64)
    mse = np.mean((original[:len(stego)] - stego)**2)
    return 10 * np.log10((32767**2) / mse) if mse > 0 else 100.0

def run_full_steganalysis(wav_path: str) -> dict:
    """
    Runs all 4 steganalysis methods on a single WAV file.
    Returns a unified results dict.

    Methods:
        1. Chi-Square Test   (from existing steganalysis.py)
        2. RS Analysis
        3. Sample Pair Analysis
        4. Histogram Analysis
    """

    print(f"\n[STEGANALYSIS] Running full battery on: {wav_path}")
    print("-" * 55)

    chi    = chi_square_test(wav_path)
    rs     = rs_analysis(wav_path)
    spa    = sample_pair_analysis(wav_path)
    hist   = histogram_analysis(wav_path)

    results = {
        "file"            : wav_path,
        "chi_square"      : chi,
        "rs_analysis"     : rs,
        "spa"             : spa,
        "histogram"       : hist,
        "detection_summary": {
            "chi_square" : chi["detected"],
            "rs_analysis": rs["detected"],
            "spa"        : spa["detected"],
            "histogram"  : hist["detected"],
            "methods_triggered": sum([
                chi["detected"],
                rs["detected"],
                spa["detected"],
                hist["detected"]
            ])
        }
    }

    _print_battery_results(results)
    return results


def _print_battery_results(results: dict):
    """Pretty-prints the full steganalysis battery results."""
    summary = results["detection_summary"]

    print(f"\n{'='*60}")
    print(f"  FULL STEGANALYSIS BATTERY RESULTS")
    print(f"  File: {results['file']}")
    print(f"{'='*60}")
    print(f"  {'Method':<28} {'Detected':>10}  Verdict")
    print(f"  {'-'*56}")

    methods = [
        ("Chi-Square Test",      results["chi_square"]),
        ("RS Analysis",          results["rs_analysis"]),
        ("Sample Pair Analysis", results["spa"]),
        ("Histogram Analysis",   results["histogram"]),
    ]

    for name, r in methods:
        flag = "YES" if r["detected"] else "NO "
        print(f"  {name:<28} {flag:>10}  {r['verdict']}")

    print(f"  {'-'*56}")
    print(f"  Methods triggered: {summary['methods_triggered']} / 4")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  COMPARISON TABLE: KEY PAPER RESULT
# ─────────────────────────────────────────────────────────────────────────────

def run_detection_matrix(
    original_wav : str,
    sequential_stego_wav  : str,
    randomized_stego_wav  : str,
    adaptive_stego_wav    : str
) -> dict:
    """
    Generates the full detection resistance matrix for the paper.

    Runs all 4 steganalysis methods against 3 embedding strategies + original.
    This is Table X in the paper — the core experimental result of Gap 2.

    Args:
        original_wav          : Clean original audio
        sequential_stego_wav  : Stego made with sequential LSB (basic)
        randomized_stego_wav  : Stego made with randomized LSB (existing method)
        adaptive_stego_wav    : Stego made with adaptive LSB (new method)

    Returns:
        matrix dict — rows = methods, columns = audio types
    """
    print("\n[MATRIX] Running detection resistance matrix...")
    print("This generates Table X for your paper.\n")

    files = {
        "Original"         : original_wav,
        "Sequential LSB"   : sequential_stego_wav,
        "Randomized LSB"   : randomized_stego_wav,
        "Adaptive LSB"     : adaptive_stego_wav,
    }

    all_results = {}
    for label, path in files.items():
        print(f"  Analyzing: {label}")
        all_results[label] = run_full_steganalysis(path)

    # Build matrix
    method_names = ["chi_square", "rs_analysis", "spa", "histogram"]
    method_labels = ["Chi-Square", "RS Analysis", "SPA", "Histogram"]

    matrix = {}
    for mname, mlabel in zip(method_names, method_labels):
        matrix[mlabel] = {}
        for label in files:
            matrix[mlabel][label] = all_results[label]["detection_summary"][mname]

    _print_detection_matrix(matrix)
    return {"matrix": matrix, "raw_results": all_results}


def _print_detection_matrix(matrix: dict):
    """Prints the detection matrix as an ASCII table."""
    cols = ["Original", "Sequential LSB", "Randomized LSB", "Adaptive LSB"]

    print(f"\n{'='*72}")
    print(f"  DETECTION RESISTANCE MATRIX")
    print(f"{'='*72}")
    print(f"  {'Method':<20}", end="")
    for col in cols:
        print(f"  {col:<16}", end="")
    print()
    print(f"  {'-'*68}")

    for method, row in matrix.items():
        print(f"  {method:<20}", end="")
        for col in cols:
            val = "DETECTED" if row[col] else "NOT DET."
            print(f"  {val:<16}", end="")
        print()

    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION: MULTI-METHOD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_steganalysis(
    results_dict : dict,
    save_path    : str = "steganalysis_matrix.png"
):
    """
    Plots a visual comparison of all 4 steganalysis methods across
    all 3 embedding modes. Generates the paper figure.

    Args:
        results_dict : Output from run_detection_matrix()
        save_path    : Where to save the PNG
    """
    matrix     = results_dict["matrix"]
    raw        = results_dict["raw_results"]
    audio_types = ["Original", "Sequential LSB", "Randomized LSB", "Adaptive LSB"]
    methods     = list(matrix.keys())

    # Build numeric grid (1 = detected, 0 = not detected)
    grid = np.zeros((len(methods), len(audio_types)))
    for i, method in enumerate(methods):
        for j, atype in enumerate(audio_types):
            grid[i, j] = 1 if matrix[method].get(atype, False) else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: Detection heatmap ──
    ax = axes[0]
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(audio_types)))
    ax.set_xticklabels(audio_types, rotation=15, ha='right', fontsize=10)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_title("Detection Resistance Matrix\n(Green = Not Detected, Red = Detected)",
                 fontsize=11, fontweight='bold')

    for i in range(len(methods)):
        for j in range(len(audio_types)):
            label = "DETECT" if grid[i, j] == 1 else "PASS"
            color = "white"
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    # ── Right: Bar chart of methods triggered per audio type ──
    ax2    = axes[1]
    counts = [sum(matrix[m].get(at, False) for m in methods) for at in audio_types]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    bars   = ax2.bar(audio_types, counts, color=colors, edgecolor='black', width=0.5)
    ax2.set_ylim(0, len(methods) + 0.5)
    ax2.set_ylabel("Detection Methods Triggered (out of 4)", fontsize=10)
    ax2.set_title("Total Detection Count per Embedding Mode", fontsize=11, fontweight='bold')
    ax2.set_xticklabels(audio_types, rotation=15, ha='right', fontsize=10)
    ax2.axhline(y=2, color='gray', linestyle='--', linewidth=1, label='50% threshold')
    ax2.legend(fontsize=9)

    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 str(count), ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    plt.suptitle("Multi-Method Steganalysis Comparison",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Steganalysis matrix graph saved to: {save_path}")
