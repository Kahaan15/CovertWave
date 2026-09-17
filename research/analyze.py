"""
analyze.py — Statistical analysis of the corrected benchmark.

Produces the numbers the paper needs and the original study never computed:

  * Clean-cover false-positive baseline (what the battery says about audio
    containing nothing at all).
  * Per-detector statistical power: true-positive rate at the calibrated 5%
    false-positive rate. Two of the four classical detectors turn out to have
    essentially none on 16-bit audio, which is a result rather than a defect.
  * Paired significance tests for every method comparison. The design is paired
    (every method sees every cover file), so McNemar's exact test is both the
    correct test and more powerful than the unpaired alternative. The original
    paper claimed "p < 0.01 in all domains" with no test of any kind in the
    codebase.
  * Effect sizes in percentage points with Wilson confidence intervals, so the
    "20% higher" ambiguity between points and relative change cannot recur.
  * Component ablation: how much each of {AES, randomisation, energy weighting}
    contributes on its own.
  * Extraction reliability and wall-clock cost per method.

Run:  python research/analyze.py
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.steganalysis import STAT_KEYS, load_thresholds

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_CSV = os.path.join(HERE, "results_main.csv")
ABLATION_CSV = os.path.join(HERE, "results_ablation.csv")
REPORT = os.path.join(HERE, "analysis_report.txt")

METHODS = ["Sequential", "Randomized", "Adaptive"]
RATES = ["1%", "5%", "10%", "25%", "50%"]

_out = []


def say(line=""):
    print(line)
    _out.append(line)


def rule(title):
    say()
    say("=" * 78)
    say(f"  {title}")
    say("=" * 78)


def wilson(successes, n, confidence=0.95):
    """Wilson score interval — behaves sensibly at proportions near 0 and 1."""
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_wins, b_wins):
    """
    Exact McNemar test on the discordant pairs.
    a_wins: cases where A survived and B did not; b_wins: the reverse.
    """
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    return float(binomtest(a_wins, n, 0.5, alternative="two-sided").pvalue)


def paired_survival_test(df, method_a, method_b, rate=None):
    """Compares two methods on the covers they both ran against."""
    subset = df[df["Method"].isin([method_a, method_b])]
    if rate is not None:
        subset = subset[subset["Payload_Rate"] == rate]

    pivot = subset.pivot_table(
        index=["Domain", "Filename", "Payload_Rate"],
        columns="Method", values="Survived", aggfunc="first",
    ).dropna()
    if method_a not in pivot or method_b not in pivot or pivot.empty:
        return None

    a = pivot[method_a].astype(int)
    b = pivot[method_b].astype(int)
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))

    return {
        "n_pairs": int(len(pivot)),
        "a_survived": int(a.sum()), "b_survived": int(b.sum()),
        "a_rate": float(a.mean()), "b_rate": float(b.mean()),
        "delta_pp": float(100 * (a.mean() - b.mean())),
        "relative_pct": float(100 * (a.mean() - b.mean()) / b.mean()) if b.mean() else float("nan"),
        "discordant_a": a_only, "discordant_b": b_only,
        "p_value": mcnemar_exact(a_only, b_only),
    }


# -----------------------------------------------------------------------------

def section_dataset(df):
    rule("1. DATASET AND RUN COUNTS")
    covers = df[["Domain", "Filename"]].drop_duplicates()
    say(f"  Cover files analysed        : {len(covers)}")
    for domain, group in covers.groupby("Domain"):
        say(f"    {domain:<14} {len(group)}")
    embedded = df[df["Method"] != "Control"]
    say(f"  Control (0% payload) runs   : {int((df['Method'] == 'Control').sum())}")
    say(f"  Embedded runs               : {len(embedded)}")
    say(f"  Total rows                  : {len(df)}")
    say(f"  Methods x rates             : {embedded['Method'].nunique()} x "
        f"{embedded['Payload_Rate'].nunique()}")


def section_detector_power(df):
    rule("2. DETECTOR CALIBRATION AND POWER")
    thresholds = load_thresholds()
    control = df[df["Method"] == "Control"]
    embedded = df[df["Method"] != "Control"]
    high = embedded[embedded["Payload_Rate"] == "50%"]

    say("  Threshold = 95th percentile of the clean-cover distribution,")
    say("  i.e. each detector is calibrated to a 5% false-positive rate.")
    say()
    say(f"  {'detector':<14}{'threshold':>12}{'FPR (clean)':>14}"
        f"{'TPR @50%':>12}{'power':>10}")
    say("  " + "-" * 62)

    for name in STAT_KEYS:
        col = f"det_{name}"
        fpr = float(control[col].mean()) if len(control) else float("nan")
        tpr = float(high[col].mean()) if len(high) else float("nan")
        spec = thresholds.get(name, {})
        if spec.get("saturated"):
            verdict = "NONE (saturated)"
        elif tpr - fpr > 0.30:
            verdict = "strong"
        elif tpr - fpr > 0.10:
            verdict = "weak"
        else:
            verdict = "NONE"
        say(f"  {name:<14}{spec.get('threshold', float('nan')):12.6f}"
            f"{100 * fpr:12.1f}%{100 * tpr:11.1f}%{verdict:>20}")

    say()
    say("  A detector whose TPR does not exceed its FPR is not detecting anything.")
    say("  Chi-square saturates on 16-bit audio: the LSB plane is already")
    say("  noise-like, so clean covers max out the pairs-of-values statistic and")
    say("  embedding has no headroom left to push it further.")


def section_control(df):
    rule("3. CLEAN-COVER BASELINE (the control the original study lacked)")
    control = df[df["Method"] == "Control"]
    if control.empty:
        say("  No control rows found.")
        return
    survived = int((control["Discriminative_Detected"] == 0).sum())
    lo, hi = wilson(survived, len(control))
    say(f"  Clean covers scoring SAFE (0 detectors fired): "
        f"{survived}/{len(control)} = {100 * survived / len(control):.1f}% "
        f"[95% CI {100 * lo:.1f}-{100 * hi:.1f}%]")
    say(f"  Mean discriminative detectors triggered      : "
        f"{control['Discriminative_Detected'].mean():.3f}")
    say()
    say("  Under the ORIGINAL battery this baseline was 63.6% 'survival' — the same")
    say("  as stego at 1-10% payload — meaning it could not separate cover from")
    say("  stego at all. A calibrated battery must sit near 100% here.")


def section_quality(df):
    rule("4. PERCEPTUAL QUALITY")
    embedded = df[df["Method"] != "Control"]
    say(f"  {'rate':>6}  " + "".join(f"{m:>26}" for m in METHODS))
    say(f"  {'':>6}  " + "".join(f"{'PSNR dB      SNR dB':>26}" for _ in METHODS))
    say("  " + "-" * 84)
    for rate in RATES:
        cells = []
        for method in METHODS:
            sub = embedded[(embedded["Method"] == method) & (embedded["Payload_Rate"] == rate)]
            cells.append(f"{sub['PSNR'].mean():13.2f}{sub['SNR'].mean():13.2f}")
        say(f"  {rate:>6}  " + "".join(cells))
    say()
    say("  PSNR is a function of how many bits were flipped, which is identical")
    say("  across the three methods by construction, so these columns agreeing is")
    say("  expected and is NOT evidence of anything about the position strategies.")

    say()
    say("  Spectral flatness deviation and LSB-plane entropy change:")
    say(f"  {'rate':>6}  " + "".join(f"{m:>24}" for m in METHODS))
    say(f"  {'':>6}  " + "".join(f"{'SF dev    dEntropy':>24}" for _ in METHODS))
    say("  " + "-" * 78)
    for rate in RATES:
        cells = []
        for method in METHODS:
            sub = embedded[(embedded["Method"] == method) & (embedded["Payload_Rate"] == rate)]
            cells.append(f"{sub['SF_Change'].mean():13.6f}{sub['Entropy_Change'].mean():11.6f}")
        say(f"  {rate:>6}  " + "".join(cells))


def section_survival(df):
    rule("5. STEGANALYSIS SURVIVAL")
    embedded = df[df["Method"] != "Control"].copy()
    say(f"  {'rate':>6}" + "".join(f"{m:>22}" for m in METHODS))
    say("  " + "-" * 72)
    for rate in RATES:
        cells = []
        for method in METHODS:
            sub = embedded[(embedded["Method"] == method) & (embedded["Payload_Rate"] == rate)]
            s = int(sub["Survived"].sum()); n = len(sub)
            lo, hi = wilson(s, n)
            cells.append(f"{100 * s / n:7.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}]")
        say(f"  {rate:>6}" + "".join(f"{c:>22}" for c in cells))
    say()
    say("  'Survived' = zero discriminative detectors fired. Brackets are 95% Wilson CIs.")


def section_significance(df):
    rule("6. SIGNIFICANCE TESTS (paired, McNemar exact)")
    embedded = df[df["Method"] != "Control"].copy()

    say("  Adaptive vs Randomized, per payload rate:")
    say(f"  {'rate':>6}{'adaptive':>11}{'randomized':>13}{'delta pp':>11}"
        f"{'relative':>11}{'p':>10}{'':>6}")
    say("  " + "-" * 68)
    for rate in RATES:
        r = paired_survival_test(embedded, "Adaptive", "Randomized", rate)
        if not r:
            continue
        mark = "**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "ns")
        say(f"  {rate:>6}{100 * r['a_rate']:10.1f}%{100 * r['b_rate']:12.1f}%"
            f"{r['delta_pp']:11.1f}{r['relative_pct']:10.1f}%{r['p_value']:10.4f}{mark:>6}")

    say()
    say("  Adaptive vs Randomized, per domain at 50% payload:")
    say(f"  {'domain':>14}{'adaptive':>11}{'randomized':>13}{'delta pp':>11}{'p':>10}{'':>6}")
    say("  " + "-" * 66)
    high = embedded[embedded["Payload_Rate"] == "50%"]
    for domain in sorted(high["Domain"].unique()):
        r = paired_survival_test(high[high["Domain"] == domain], "Adaptive", "Randomized")
        if not r:
            continue
        mark = "**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "ns")
        say(f"  {domain:>14}{100 * r['a_rate']:10.1f}%{100 * r['b_rate']:12.1f}%"
            f"{r['delta_pp']:11.1f}{r['p_value']:10.4f}{mark:>6}")

    say()
    say("  Pooled over all payload rates:")
    for a, b in (("Adaptive", "Randomized"), ("Adaptive", "Sequential"),
                 ("Randomized", "Sequential")):
        r = paired_survival_test(embedded, a, b)
        if not r:
            continue
        mark = "**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "ns")
        say(f"    {a:<11} vs {b:<11} delta {r['delta_pp']:+6.1f} pp "
            f"({r['relative_pct']:+.1f}% relative), n={r['n_pairs']}, "
            f"p={r['p_value']:.4f} {mark}")

    say()
    say("  Detector statistics compared directly (Mann-Whitney U on rs_diff, 50%):")
    for a, b in (("Adaptive", "Randomized"), ("Adaptive", "Sequential")):
        xa = high[high["Method"] == a]["stat_rs_analysis"].dropna()
        xb = high[high["Method"] == b]["stat_rs_analysis"].dropna()
        if len(xa) and len(xb):
            u, p = mannwhitneyu(xa, xb, alternative="two-sided")
            say(f"    {a} ({xa.mean():.5f}) vs {b} ({xb.mean():.5f}): p={p:.3g}")


def section_reliability(df):
    rule("7. EXTRACTION RELIABILITY AND COST")
    embedded = df[df["Method"] != "Control"]
    say(f"  {'method':<14}{'extraction OK':>16}{'mean BER':>12}"
        f"{'encode s':>11}{'decode s':>11}")
    say("  " + "-" * 66)
    for method in METHODS:
        sub = embedded[embedded["Method"] == method]
        ok = int(sub["Extraction_OK"].sum())
        say(f"  {method:<14}{ok:>7}/{len(sub):<8}{sub['BER'].mean():12.6f}"
            f"{sub['Encode_Seconds'].mean():11.3f}{sub['Decode_Seconds'].mean():11.3f}")
    say()
    say("  Before the energy-profile fix the adaptive decoder failed on 30% of the")
    say("  corpus at 1-bit (43% at 2-bit, 55% at 4-bit) because encoder and decoder")
    say("  derived their weights from different signals.")


def section_ablation():
    if not os.path.exists(ABLATION_CSV):
        return
    rule("8. COMPONENT ABLATION  {plaintext, AES} x {sequential, uniform, adaptive}")
    ab = pd.read_csv(ABLATION_CSV)
    ab["Survived"] = (ab["Discriminative_Detected"] == 0).astype(int)

    for rate in sorted(ab["Payload_Rate"].unique(),
                       key=lambda r: int(r.rstrip("%"))):
        sub = ab[ab["Payload_Rate"] == rate]
        say()
        say(f"  Payload {rate} — survival rate")
        say(f"  {'strategy':<14}{'plaintext':>14}{'AES':>14}{'crypto delta':>16}")
        say("  " + "-" * 58)
        for strategy in ("sequential", "uniform", "adaptive"):
            plain = sub[(sub["Strategy"] == strategy) & (sub["Encrypted"] == 0)]["Survived"]
            aes = sub[(sub["Strategy"] == strategy) & (sub["Encrypted"] == 1)]["Survived"]
            if plain.empty or aes.empty:
                continue
            say(f"  {strategy:<14}{100 * plain.mean():13.1f}%{100 * aes.mean():13.1f}%"
                f"{100 * (aes.mean() - plain.mean()):15.1f}pp")

        aes_only = sub[sub["Encrypted"] == 1]
        base = aes_only[aes_only["Strategy"] == "sequential"]["Survived"].mean()
        uni = aes_only[aes_only["Strategy"] == "uniform"]["Survived"].mean()
        ada = aes_only[aes_only["Strategy"] == "adaptive"]["Survived"].mean()
        say()
        say(f"    contribution of randomisation  (sequential -> uniform): "
            f"{100 * (uni - base):+.1f} pp")
        say(f"    contribution of energy weighting (uniform -> adaptive): "
            f"{100 * (ada - uni):+.1f} pp")


def main():
    if not os.path.exists(MAIN_CSV):
        print(f"[ERROR] {MAIN_CSV} not found. Run research/benchmark.py first.")
        return

    df = pd.read_csv(MAIN_CSV)
    df["Survived"] = (df["Discriminative_Detected"] == 0).astype(int)

    say("COVERTWAVE — CORRECTED BENCHMARK ANALYSIS")

    section_dataset(df)
    section_detector_power(df)
    section_control(df)
    section_quality(df)
    section_survival(df)
    section_significance(df)
    section_reliability(df)
    section_ablation()

    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(_out) + "\n")
    print(f"\n[SAVED] {REPORT}")


if __name__ == "__main__":
    main()
