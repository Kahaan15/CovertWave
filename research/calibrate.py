"""
calibrate.py — Derives detection thresholds from clean cover audio.

Each detector in core/steganalysis.py produces a continuous statistic. A verdict
needs a threshold, and the defensible way to set one is to fix the false-positive
rate against audio that is known to contain nothing.

For every clean cover we compute all four statistics, then set each threshold at
the (100 - TARGET_FPR)th percentile of that distribution. By construction each
detector then fires on TARGET_FPR% of clean covers, so "detected" means
"statistically distinguishable from clean audio at a 5% false-positive rate" —
a claim with a defined error rate attached, rather than a hand-picked constant.

This also produces the clean-cover baseline the original study never had. Without
it there is no way to tell whether a detector is responding to the embedding or
merely to the audio: the previous battery scored clean covers at 63.6% "survival"
and stego at 1-10% payload at 62-65%, which is the signature of a detector that
is measuring the cover, not the payload.

Run:  python research/calibrate.py
"""

import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio_io import UnreadableAudio, read_mono
from core.steganalysis import (
    STAT_KEYS,
    THRESHOLD_FILE,
    chi_square_test,
    histogram_analysis,
    rs_analysis,
    sample_pair_analysis,
)

TARGET_FPR = 5.0          # percent of clean covers each detector may flag
DOMAINS = ["ambient", "instrumental", "short_voice", "speech"]


def collect_clean_statistics(dataset_root="dataset"):
    """Computes all four statistics for every readable cover file."""
    rows = []
    for domain in DOMAINS:
        domain_path = os.path.join(dataset_root, domain)
        if not os.path.isdir(domain_path):
            continue
        for filename in sorted(os.listdir(domain_path)):
            if not filename.endswith(".wav"):
                continue
            path = os.path.join(domain_path, filename)
            try:
                samples = read_mono(path)
            except (UnreadableAudio, ValueError) as exc:
                print(f"  [SKIP] {filename}: {exc}")
                continue

            row = {"domain": domain, "filename": filename}
            row["chi_square"] = chi_square_test(samples=samples)["equalized_block_fraction"]
            row["rs_analysis"] = rs_analysis(samples=samples)["rs_diff"]
            row["spa"] = sample_pair_analysis(samples=samples)["beta"]
            row["histogram"] = histogram_analysis(samples=samples)["pov_equalization"]
            rows.append(row)
            print(f"  {domain}/{filename}")
    return rows


def derive_thresholds(rows, target_fpr=TARGET_FPR):
    """Sets each threshold at the (100 - target_fpr)th percentile of clean values."""
    thresholds = {}
    distribution = {}

    for detector, stat_key in STAT_KEYS.items():
        values = np.array([row[detector] for row in rows], dtype=np.float64)
        cutoff = float(np.percentile(values, 100.0 - target_fpr))

        # A statistic whose 95th percentile already sits at its ceiling is
        # saturated on this medium: clean covers max it out, so there is no
        # headroom for embedding to push it further and the detector carries no
        # discriminative power. Flag it rather than pretend it is calibrated.
        saturated = bool(cutoff >= float(np.max(values)))

        thresholds[detector] = {
            "statistic": stat_key,
            "threshold": cutoff,
            "saturated": saturated,
        }
        distribution[detector] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "realised_fpr_percent": float(100.0 * np.mean(values > cutoff)),
        }

    return thresholds, distribution


def main():
    print("=" * 72)
    print("  CALIBRATION - clean-cover baseline")
    print("=" * 72)

    rows = collect_clean_statistics()
    if not rows:
        print("[ERROR] No readable cover audio found under dataset/.")
        return

    thresholds, distribution = derive_thresholds(rows)

    payload = {
        "target_false_positive_rate_percent": TARGET_FPR,
        "n_clean_covers": len(rows),
        "thresholds": thresholds,
        "clean_distribution": distribution,
        "note": (
            "Thresholds are the 95th percentile of each statistic over the clean "
            "cover set, giving a nominal 5% false-positive rate per detector."
        ),
    }

    with open(THRESHOLD_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\n  Calibrated on {len(rows)} clean covers -> {THRESHOLD_FILE}\n")
    header = (f"  {'detector':<14}{'mean':>12}{'std':>12}{'p95=thresh':>14}"
              f"{'max':>12}{'FPR%':>8}{'status':>18}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for detector in STAT_KEYS:
        d = distribution[detector]
        status = "SATURATED" if thresholds[detector]["saturated"] else "calibrated"
        print(f"  {detector:<14}{d['mean']:12.6f}{d['std']:12.6f}"
              f"{thresholds[detector]['threshold']:14.6f}{d['max']:12.6f}"
              f"{d['realised_fpr_percent']:8.1f}{status:>18}")

    # Save the per-file clean statistics too - this is the false-positive
    # baseline table the paper needs.
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "clean_cover_baseline.csv")
    import csv as _csv
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(
            handle, fieldnames=["domain", "filename"] + list(STAT_KEYS)
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Per-file clean statistics -> {csv_path}")


if __name__ == "__main__":
    main()
