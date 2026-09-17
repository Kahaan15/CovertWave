"""
steganalysis.py — Corrected four-method steganalysis battery.

Every detector here returns a *continuous statistic* plus a verdict. The verdict
comes from a threshold calibrated on clean cover audio (see research/calibrate.py),
not from a hand-chosen constant. "Detected" therefore means "statistically
distinguishable from the clean-cover distribution", which is a claim that can be
defended, rather than "exceeded a number someone typed".

Why the rewrite — measured behaviour of the previous implementation on the 44
clean, unmodified covers in dataset/:

    Sample Pair Analysis   flagged 44/44   beta = (W-Z)/(2n)+0.5 is not the
                                           Dumitrescu estimator; it returns ~0.5
                                           for any signal with random LSBs
    Histogram              flagged 16/44   compared adjacent bins of a 256-bin
                                           histogram over +/-32768, so each bin
                                           spanned ~256 values and the value-pair
                                           effect it claimed to measure was invisible
    Chi-Square             flagged  0/44   threshold avg_p > 0.9 against an
                                           observed ~0.50; could never fire
    RS Analysis            flagged  0/44   the only one carrying real signal

Clean covers scored 63.6% "survival" under that battery and stego at 1-10%
payload scored 62-65%, i.e. it could not separate cover from stego at all.
"""

import json
import os

import numpy as np
from scipy.stats import chi2 as chi2_dist

from .audio_io import read_mono

THRESHOLD_FILE = os.path.join(os.path.dirname(__file__), "thresholds.json")

# Fallbacks used only if calibration has not been run yet.
_DEFAULT_THRESHOLDS = {
    "chi_square": {"statistic": "equalized_block_fraction", "threshold": 0.15},
    "rs_analysis": {"statistic": "rs_diff", "threshold": 0.05},
    "spa": {"statistic": "beta", "threshold": 0.005},
    "histogram": {"statistic": "pov_equalization", "threshold": 0.60},
}


def load_thresholds() -> dict:
    """Loads calibrated thresholds, falling back to defaults if absent."""
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)["thresholds"]
    return _DEFAULT_THRESHOLDS


# -----------------------------------------------------------------------------
#  1. CHI-SQUARE - pairs-of-values attack, computed per block
# -----------------------------------------------------------------------------

def chi_square_test(wav_path=None, samples=None, block_size: int = 65536) -> dict:
    """
    Classical PoV chi-square attack (Westfeld & Pfitzmann), applied per block.

    For each pair of values (2k, 2k+1) the counts should be equalized by LSB
    embedding. The statistic compares observed n(2k) against the pair mean
    (n(2k)+n(2k+1))/2. A block whose p-value approaches 1 has suspiciously
    equalized pairs.

    Run per block rather than globally because that is what exposes SEQUENTIAL
    embedding: a contiguous run of equalized blocks at the head of the file
    followed by untouched blocks. A single global statistic averages that away.
    """
    if samples is None:
        samples = read_mono(wav_path)
    samples = samples.astype(np.int64)

    n_blocks = max(1, len(samples) // block_size)
    p_values = []

    for i in range(n_blocks):
        block = samples[i * block_size:(i + 1) * block_size]
        if len(block) < 1024:
            continue

        shifted = block - block.min()
        counts = np.bincount(shifted)
        if len(counts) % 2:
            counts = np.append(counts, 0)

        even = counts[0::2].astype(np.float64)
        odd = counts[1::2].astype(np.float64)
        expected = (even + odd) / 2.0

        usable = expected >= 5
        dof = int(np.sum(usable)) - 1
        if dof < 1:
            continue

        stat = float(np.sum((even[usable] - expected[usable]) ** 2 / expected[usable]))
        p_values.append(float(chi2_dist.sf(stat, dof)))

    if not p_values:
        return {
            "method": "Chi-Square Test", "equalized_block_fraction": 0.0,
            "mean_p_value": 0.0, "n_blocks": 0, "max_run": 0,
            "verdict": "INCONCLUSIVE", "detected": False,
        }

    p_values = np.array(p_values)
    equalized = p_values > 0.95
    fraction = float(np.mean(equalized))

    # Longest contiguous run of equalized blocks - the sequential signature.
    max_run = run = 0
    for flag in equalized:
        run = run + 1 if flag else 0
        max_run = max(max_run, run)

    return {
        "method": "Chi-Square Test",
        "equalized_block_fraction": round(fraction, 6),
        "mean_p_value": round(float(np.mean(p_values)), 6),
        "n_blocks": int(len(p_values)),
        "max_run": int(max_run),
        "verdict": "", "detected": False,
    }


# -----------------------------------------------------------------------------
#  2. RS ANALYSIS - Fridrich, Goljan & Du
# -----------------------------------------------------------------------------

def rs_analysis(wav_path=None, samples=None, group_size: int = 4) -> dict:
    """
    Regular/Singular analysis with the standard mask [0,1,0,1] and its negation.
    Vectorized; integer arithmetic throughout (the previous version round-tripped
    through float64, which is lossy and needless here).
    """
    if samples is None:
        samples = read_mono(wav_path)
    samples = samples.astype(np.int64)

    n_groups = len(samples) // group_size
    if n_groups < 2:
        raise ValueError("Audio too short for RS analysis.")

    groups = samples[:n_groups * group_size].reshape(n_groups, group_size)
    mask = np.tile([0, 1], group_size // 2)[:group_size].astype(bool)

    def discriminant(block):
        return np.sum(np.abs(np.diff(block, axis=1)), axis=1)

    d_orig = discriminant(groups)

    flipped = groups.copy()
    flipped[:, mask] ^= 1                        # F_1
    d_flip = discriminant(flipped)

    shifted = groups.copy()
    vals = shifted[:, mask]
    even = (vals % 2) == 0
    vals[even] -= 1                              # F_-1
    vals[~even] += 1
    shifted[:, mask] = vals
    d_shift = discriminant(shifted)

    rm = float(np.mean(d_flip > d_orig))
    sm = float(np.mean(d_flip < d_orig))
    rn = float(np.mean(d_shift > d_orig))
    sn = float(np.mean(d_shift < d_orig))

    d0, d1 = rm - rn, sm - sn
    rate = abs(d0 / (d0 - d1)) if abs(d0 - d1) > 1e-12 else 0.0

    return {
        "method": "RS Analysis",
        "rm": round(rm, 6), "sm": round(sm, 6),
        "rn": round(rn, 6), "sn": round(sn, 6),
        "rs_diff": round(abs((rm - sm) - (rn - sn)), 6),
        "estimated_rate": round(min(max(rate, 0.0), 1.0), 6),
        "verdict": "", "detected": False,
    }


# -----------------------------------------------------------------------------
#  3. SAMPLE PAIR ANALYSIS - Dumitrescu, Wu & Wang (2003)
# -----------------------------------------------------------------------------

def sample_pair_analysis(wav_path=None, samples=None) -> dict:
    """
    The actual SPA estimator, solving the quadratic

        0.5 (W + Z) b^2 + (2X - W) b + (X - Y) = 0

    for the modification rate b, where over disjoint sample pairs (u, v):

        W = pairs differing only in the LSB      Z = pairs with u == v
        X = pairs with (v even, u < v) or (v odd, u > v)
        Y = pairs with (v even, u > v) or (v odd, u < v)

    Note on audio: W + Z is only ~3% of pairs in 16-bit audio (adjacent samples
    rarely land in the same PoV bucket), against a large majority in 8-bit
    imagery, so the estimator is far less sensitive here than its image-domain
    reputation suggests. It stays monotonic in the true embedding rate and the
    clean-cover distribution is tightly centred on zero, so it discriminates well
    once thresholded against that baseline - but the raw beta it reports on audio
    understates the true rate by roughly an order of magnitude and must not be
    read as an absolute payload estimate.
    """
    if samples is None:
        samples = read_mono(wav_path)
    samples = samples.astype(np.int64)

    u = samples[0::2]
    v = samples[1::2]
    n = min(len(u), len(v))
    if n < 2:
        raise ValueError("Audio too short for sample pair analysis.")
    u, v = u[:n], v[:n]

    W = int(np.sum(((u >> 1) == (v >> 1)) & (u != v)))
    Z = int(np.sum(u == v))

    v_even = (v % 2) == 0
    X = int(np.sum((v_even & (u < v)) | (~v_even & (u > v))))
    Y = int(np.sum((v_even & (u > v)) | (~v_even & (u < v))))

    a = 0.5 * (W + Z)
    b = 2.0 * X - W
    c = float(X - Y)

    if abs(a) < 1e-12:
        beta = 0.0 if abs(b) < 1e-12 else -c / b
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            beta = 0.0
        else:
            roots = [r.real for r in np.roots([a, b, c]) if abs(r.imag) < 1e-9]
            candidates = [r for r in roots if -0.5 <= r <= 1.0]
            beta = min(candidates, key=abs) if candidates else 0.0

    return {
        "method": "Sample Pair Analysis",
        "total_pairs": int(n),
        "W": W, "Z": Z, "X": X, "Y": Y,
        "beta": round(float(beta), 8),
        "verdict": "", "detected": False,
    }


# -----------------------------------------------------------------------------
#  4. HISTOGRAM - value-pair (PoV) equalization
# -----------------------------------------------------------------------------

def histogram_analysis(wav_path=None, samples=None) -> dict:
    """
    Measures equalization of the value pairs (2k, 2k+1) directly on the sample
    histogram, at value resolution.

    LSB embedding drives n(2k) and n(2k+1) together. The statistic is the
    count-weighted mean of

        1 - |n(2k) - n(2k+1)| / (n(2k) + n(2k+1))

    which runs from 0 (pairs maximally unbalanced) to 1 (fully equalized).

    The previous implementation binned into 256 bins across the full +/-32768
    range, so one bin covered ~256 distinct values and the effect being measured
    was averaged out of existence before the comparison happened.
    """
    if samples is None:
        samples = read_mono(wav_path)
    samples = samples.astype(np.int64)

    shifted = samples - samples.min()
    counts = np.bincount(shifted)
    if len(counts) % 2:
        counts = np.append(counts, 0)

    even = counts[0::2].astype(np.float64)
    odd = counts[1::2].astype(np.float64)
    total = even + odd

    usable = total >= 20
    if not np.any(usable):
        raise ValueError("Too few populated value pairs for histogram analysis.")

    balance = 1.0 - np.abs(even[usable] - odd[usable]) / total[usable]
    weights = total[usable]
    equalization = float(np.average(balance, weights=weights))

    return {
        "method": "Histogram Analysis",
        "pov_equalization": round(equalization, 6),
        "pairs_used": int(np.sum(usable)),
        "verdict": "", "detected": False,
    }


# -----------------------------------------------------------------------------
#  BATTERY
# -----------------------------------------------------------------------------

STAT_KEYS = {
    "chi_square": "equalized_block_fraction",
    "rs_analysis": "rs_diff",
    "spa": "beta",
    "histogram": "pov_equalization",
}


def run_full_steganalysis(wav_path=None, samples=None, thresholds=None, verbose=False) -> dict:
    """
    Runs all four detectors on one file and applies the calibrated thresholds.
    Reads the audio once and shares it, instead of re-decoding it per method.
    """
    if samples is None:
        samples = read_mono(wav_path)
    if thresholds is None:
        thresholds = load_thresholds()

    results = {
        "chi_square": chi_square_test(samples=samples),
        "rs_analysis": rs_analysis(samples=samples),
        "spa": sample_pair_analysis(samples=samples),
        "histogram": histogram_analysis(samples=samples),
    }

    for name, result in results.items():
        key = STAT_KEYS[name]
        spec = thresholds.get(name, {})
        limit = spec.get("threshold")
        if limit is None or result.get("verdict") == "INCONCLUSIVE":
            continue

        # A saturated detector is reported, but never counted as a detection:
        # clean covers already sit at its ceiling, so a "hit" would carry no
        # information. See research/calibrate.py.
        if spec.get("saturated"):
            result["saturated"] = True
            result["threshold"] = limit
            result["detected"] = False
            result["verdict"] = (
                f"NON-DISCRIMINATIVE ({key}={result[key]:.6g}; clean covers "
                f"saturate this statistic on 16-bit audio)"
            )
            continue

        detected = bool(result[key] > limit)
        result["detected"] = detected
        result["threshold"] = limit
        result["verdict"] = (
            f"STEGO DETECTED ({key}={result[key]:.6g} > {limit:.6g})" if detected
            else f"CLEAN ({key}={result[key]:.6g} <= {limit:.6g})"
        )

    triggered = sum(results[name]["detected"] for name in STAT_KEYS)
    discriminative = [n for n in STAT_KEYS if not results[n].get("saturated")]

    results["file"] = wav_path
    results["detection_summary"] = {name: results[name]["detected"] for name in STAT_KEYS}
    results["detection_summary"]["methods_triggered"] = triggered
    results["detection_summary"]["discriminative_methods"] = len(discriminative)
    results["detection_summary"]["discriminative_triggered"] = sum(
        results[n]["detected"] for n in discriminative
    )

    if verbose:
        print(f"\n[STEGANALYSIS] {wav_path}")
        for name in STAT_KEYS:
            r = results[name]
            print(f"  {r['method']:<24} {'YES' if r['detected'] else 'NO':>4}  {r['verdict']}")
        print(f"  Methods triggered: {triggered}/4")

    return results
