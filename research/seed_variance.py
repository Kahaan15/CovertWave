"""
seed_variance.py — How much of the reported result is run-to-run noise?

The benchmark embeds each condition once, with one password. Positions are drawn
deterministically from that password, so a single run measures one realisation of
a stochastic method. Two independent runs of the *identical* condition (adaptive,
AES, 50% payload) in the main sweep and in the ablation gave 56.8% and 54.5%
survival, with 7 of 44 files changing verdict, so the realisation matters.

This script quantifies it. For each repeat a different password is used, which
re-draws both the AES key and the whole position set, and the survival rate over
all covers is recomputed. Reporting the spread across repeats converts an
unstated source of uncertainty into a measured one.

Only the two discriminative detectors are evaluated (RS, SPA); the two saturated
pairs-of-values tests contribute nothing and cost time.

Run:  python research/seed_variance.py
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio_io import UnreadableAudio, read_mono
from core.embedding import embed
from core.steganalysis import load_thresholds, rs_analysis, sample_pair_analysis

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "seed_variance.csv")
REPORT = os.path.join(HERE, "seed_variance_report.txt")

DOMAINS = ["ambient", "instrumental", "short_voice", "speech"]
METHODS = [("Sequential", "sequential"), ("Randomized", "uniform"), ("Adaptive", "adaptive")]
RATES = [0.50, 0.25]
N_REPEATS = 10
LSB_BITS = 1

_out = []


def say(line=""):
    print(line)
    _out.append(line)


def covers(dataset_root="dataset"):
    out = []
    for domain in DOMAINS:
        path = os.path.join(dataset_root, domain)
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            if not name.endswith(".wav"):
                continue
            full = os.path.join(path, name)
            try:
                n = len(read_mono(full))
            except (UnreadableAudio, ValueError):
                continue
            out.append((domain, name, full, n // 8))
    return out


def survived(samples, thresholds):
    """Zero discriminative detectors firing, as in the main benchmark."""
    rs = rs_analysis(samples=samples)["rs_diff"]
    spa = sample_pair_analysis(samples=samples)["beta"]
    hit = (rs > thresholds["rs_analysis"]["threshold"]) or \
          (spa > thresholds["spa"]["threshold"])
    return 0 if hit else 1


def main():
    thresholds = load_thresholds()
    files = covers()
    tmp = os.path.join(HERE, f"_var_{os.getpid()}.wav")

    say("=" * 78)
    say("  RUN-TO-RUN VARIANCE OF THE REPORTED SURVIVAL RATES")
    say("=" * 78)
    say(f"  {len(files)} covers x {len(METHODS)} methods x {len(RATES)} payloads"
        f" x {N_REPEATS} passwords")
    say(f"  Each repeat re-draws the AES key and the entire position set.")
    say()

    rows = []
    t0 = time.perf_counter()
    for rate in RATES:
        label = f"{int(rate * 100)}%"
        for method, strategy in METHODS:
            for rep in range(N_REPEATS):
                password = f"variance_probe_seed_{rep:02d}"
                s = 0
                n = 0
                for domain, name, path, cap in files:
                    n_bytes = max(10, int(cap * rate) - 40)
                    try:
                        embed(path, tmp, "X" * n_bytes, password, LSB_BITS, strategy)
                        s += survived(read_mono(tmp), thresholds)
                        n += 1
                    except Exception:
                        pass
                    finally:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                rate_pct = 100.0 * s / n if n else float("nan")
                rows.append({"payload": label, "method": method, "repeat": rep,
                             "n_files": n, "survived": s,
                             "survival_pct": round(rate_pct, 4)})
                print(f"  {label:>4} {method:11s} seed {rep:2d}: "
                      f"{s:2d}/{n} = {rate_pct:5.1f}%   "
                      f"[{(time.perf_counter()-t0)/60:.0f} min]")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    say()
    say(f"  {'payload':>8}{'method':>13}{'mean':>9}{'sd':>7}{'min':>7}{'max':>7}{'range':>8}")
    say("  " + "-" * 60)
    stats = {}
    for rate in RATES:
        label = f"{int(rate * 100)}%"
        for method, _ in METHODS:
            v = np.array([r["survival_pct"] for r in rows
                          if r["payload"] == label and r["method"] == method])
            stats[(label, method)] = v
            say(f"  {label:>8}{method:>13}{v.mean():8.1f}%{v.std(ddof=1):7.2f}"
                f"{v.min():7.1f}{v.max():7.1f}{v.max()-v.min():8.1f}")

    say()
    say("  ADAPTIVE vs RANDOMIZED, across repeats:")
    for rate in RATES:
        label = f"{int(rate * 100)}%"
        a = stats[(label, "Adaptive")]
        u = stats[(label, "Randomized")]
        d = a - u                       # paired by seed
        say(f"    {label:>4}: delta = {d.mean():+.1f} pp"
            f"  (sd {d.std(ddof=1):.2f}, range {d.min():+.1f} to {d.max():+.1f})")

    say()
    say("  The benchmark reports a single realisation per condition. The spread")
    say("  above is the uncertainty that figure carries beyond the Wilson")
    say("  interval, which accounts for variation across covers but not for the")
    say("  stochastic draw itself.")

    with open(REPORT, "w", encoding="utf-8") as h:
        h.write("\n".join(_out) + "\n")
    print(f"\n[SAVED] {OUT_CSV}\n[SAVED] {REPORT}")


if __name__ == "__main__":
    main()
