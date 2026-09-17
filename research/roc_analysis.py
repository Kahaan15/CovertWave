"""
roc_analysis.py — Threshold-free evaluation of the steganalysis battery.

The benchmark records a binary verdict per run, obtained by comparing each
detector's statistic against a threshold calibrated to a 5% false-positive rate.
That is defensible, but it collapses a continuous statistic to one operating
point and hides how the detector behaves everywhere else.

This script recovers the full picture from data already collected. Every run
stored its raw detector statistics (the stat_* columns), and the 0%-payload
control rows provide genuine negatives measured on the same recordings, so ROC
curves can be computed without re-running anything.

Two figures of merit per (detector, method, payload):

    AUC   area under the ROC curve; 0.5 is chance, 1.0 is perfect separation.
    P_E   min over thresholds of (P_FA + P_MD) / 2, the conventional
          steganalysis error. 0.5 means undetectable, 0 means always caught.

P_E is the one to quote. It is threshold-free, it is what the steganalysis
literature reports, and it makes these results directly comparable with other
papers in a way that a bespoke "survival rate" never can.

Run:  python research/roc_analysis.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.steganalysis import STAT_KEYS, load_thresholds

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_CSV = os.path.join(HERE, "results_main.csv")
OUT_CSV = os.path.join(HERE, "roc_results.csv")
REPORT = os.path.join(HERE, "roc_report.txt")

METHODS = ["Sequential", "Randomized", "Adaptive"]
RATES = ["1%", "5%", "10%", "25%", "50%"]
DETECTOR_LABELS = {
    "chi_square": "Chi-Square",
    "rs_analysis": "RS Analysis",
    "spa": "Sample Pair",
    "histogram": "Histogram",
}

_out = []


def say(line=""):
    print(line)
    _out.append(line)


def p_error(labels, scores):
    """min_t (P_FA(t) + P_MD(t)) / 2."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.min((fpr + (1.0 - tpr)) / 2.0))


def detector_scores(df, detector, method, rate):
    """
    Negatives from the clean-cover control, positives from the stego runs.

    Restricted to recordings present in both, so the comparison is paired and a
    file that failed one arm cannot skew the other.
    """
    column = f"stat_{detector}"
    control = df[df.Method == "Control"][["Filename", column]]
    stego = df[(df.Method == method) & (df.Payload_Rate == rate)][["Filename", column]]

    shared = set(control.Filename) & set(stego.Filename)
    control = control[control.Filename.isin(shared)]
    stego = stego[stego.Filename.isin(shared)]

    scores = np.concatenate([control[column].to_numpy(), stego[column].to_numpy()])
    labels = np.concatenate([np.zeros(len(control)), np.ones(len(stego))])
    return labels, scores, len(shared)


def main():
    if not os.path.exists(MAIN_CSV):
        print(f"[ERROR] {MAIN_CSV} not found. Run research/benchmark.py first.")
        return

    df = pd.read_csv(MAIN_CSV)
    thresholds = load_thresholds()
    rows = []

    say("=" * 78)
    say("  THRESHOLD-FREE DETECTOR EVALUATION (ROC)")
    say("=" * 78)
    say("  Negatives: 0%-payload control runs on the same recordings.")
    say("  P_E = min average decision error. 0.5 = undetectable, 0 = always caught.")

    for detector in STAT_KEYS:
        saturated = thresholds.get(detector, {}).get("saturated", False)
        say()
        say(f"  {DETECTOR_LABELS[detector]}"
            + ("   [saturated on 16-bit audio — no discriminative power]" if saturated else ""))
        say(f"  {'payload':>9}" + "".join(f"{m:>24}" for m in METHODS))
        say(f"  {'':>9}" + "".join(f"{'AUC':>12}{'P_E':>12}" for _ in METHODS))
        say("  " + "-" * 81)

        for rate in RATES:
            cells = []
            for method in METHODS:
                labels, scores, n = detector_scores(df, detector, method, rate)
                if len(set(labels.tolist())) < 2 or np.all(scores == scores[0]):
                    cells.append(f"{'--':>12}{'--':>12}")
                    continue
                auc = roc_auc_score(labels, scores)
                pe = p_error(labels, scores)
                cells.append(f"{auc:12.3f}{pe:12.3f}")
                rows.append({
                    "detector": detector, "method": method, "payload": rate,
                    "n_files": n, "auc": round(float(auc), 5),
                    "p_e": round(pe, 5),
                })
            say(f"  {rate:>9}" + "".join(cells))

    # ---- the comparison that matters ----
    say()
    say("=" * 78)
    say("  ADAPTIVE vs RANDOMIZED — the discriminative detectors only")
    say("=" * 78)
    live = [d for d in STAT_KEYS if not thresholds.get(d, {}).get("saturated", False)]
    live = [d for d in live if d in ("rs_analysis", "spa")]

    for detector in live:
        say()
        say(f"  {DETECTOR_LABELS[detector]}: P_E (higher = better hidden)")
        say(f"  {'payload':>9}{'Randomized':>14}{'Adaptive':>12}{'gain':>10}"
            f"{'Mann-Whitney p':>18}")
        say("  " + "-" * 63)
        for rate in RATES:
            l_r, s_r, _ = detector_scores(df, detector, "Randomized", rate)
            l_a, s_a, _ = detector_scores(df, detector, "Adaptive", rate)
            if len(set(l_r.tolist())) < 2:
                continue
            pe_r, pe_a = p_error(l_r, s_r), p_error(l_a, s_a)

            stego_r = s_r[l_r == 1]
            stego_a = s_a[l_a == 1]
            _, p = mannwhitneyu(stego_a, stego_r, alternative="less")
            mark = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            say(f"  {rate:>9}{pe_r:14.3f}{pe_a:12.3f}{pe_a - pe_r:+10.3f}"
                f"{p:14.4f} {mark:>3}")

    say()
    say("  The p-value tests whether the adaptive method yields SMALLER detector")
    say("  statistics than uniform randomised embedding on the same recordings —")
    say("  a direct test on the continuous statistic, with no threshold involved.")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(_out) + "\n")
    print(f"\n[SAVED] {OUT_CSV}")
    print(f"[SAVED] {REPORT}")


if __name__ == "__main__":
    main()
