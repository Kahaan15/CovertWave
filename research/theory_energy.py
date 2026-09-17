"""
theory_energy.py — Why energy-weighted placement reduces detectability.

The paper asserts that embedding into high-energy frames is safer, and justifies
it by appeal to psychoacoustic masking. Masking explains *inaudibility*. It says
nothing about *statistical detectability*, and a reviewer correctly objected that
the rationale for the RMS weighting is not established. This script supplies the
missing argument and measures it.

THE CLAIM

    Both RS analysis and Sample Pair Analysis are built on comparisons between
    adjacent samples. Write d_i = x_{i+1} - x_i.

    RS classifies a group by whether flipping the LSBs of a masked subset
    increases or decreases f(G) = sum |d_i|. SPA classifies a pair (u, v) by
    which of the sets W, Z, X, Y it falls in, all of which are defined by u - v
    and the parities of u and v.

    LSB substitution perturbs a sample by at most one quantisation step, so it
    perturbs each d_i by at most 2. That perturbation can only change a
    comparison when |d_i| is itself of order 1 — when the two samples already sit
    in the same or an adjacent value pair. Where |d_i| is large, a change of 2
    cannot alter the sign of the comparison, and the detector's classification of
    that group is invariant to the embedding.

    In a quiet frame, adjacent samples differ by 0 or 1 and almost every group is
    fragile. In a loud frame, adjacent samples differ by hundreds and almost no
    group is. Detectability per embedded bit is therefore a decreasing function
    of local signal energy, which is exactly what weighting embedding
    probabilities by frame RMS exploits.

    Prediction: flipping a FIXED number of bits confined to a high-energy decile
    perturbs the detector statistics far less than flipping the same number of
    bits confined to a low-energy decile.

THE MEASUREMENT

    For each cover, samples are sorted by the RMS energy of their 512-sample
    frame and split into ten equal-population deciles. Within one decile, K
    randomly chosen samples have their LSB flipped; every other sample is left
    alone. The resulting shift in rs_diff and in the SPA beta estimate is
    recorded. K is identical across deciles, so the comparison is per embedded
    bit by construction.

    This also explains the ablation result the benchmark produced but could not
    account for: energy weighting contributed +25.0 pp of survival while
    randomisation contributed -11.4 pp. Randomisation moves bits without regard
    to fragility; energy weighting moves them away from it.

Run:  python research/theory_energy.py
"""

import csv
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adaptive import compute_frame_energy
from core.audio_io import UnreadableAudio, read_mono
from core.steganalysis import rs_analysis, sample_pair_analysis

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "theory_energy.csv")
REPORT = os.path.join(HERE, "theory_energy_report.txt")

DOMAINS = ["ambient", "instrumental", "short_voice", "speech"]
FRAME = 512
N_DECILES = 10
FLIPS_PER_DECILE = 20000
SEED = 12345

_out = []


def say(line=""):
    print(line)
    _out.append(line)


def sample_energy(samples, frame_size=FRAME):
    """Per-sample local energy: every sample inherits its frame's RMS."""
    energy = compute_frame_energy(samples, frame_size)
    per_sample = np.repeat(energy, frame_size)
    if len(per_sample) < len(samples):          # trailing partial frame
        pad = len(samples) - len(per_sample)
        tail = samples[len(per_sample):].astype(np.float64)
        rms = float(np.sqrt(np.mean(tail ** 2))) if pad else 0.0
        per_sample = np.concatenate([per_sample, np.full(pad, rms)])
    return per_sample[:len(samples)]


def decile_assignment(energy, n_deciles=N_DECILES):
    """Equal-population deciles by local energy. Returns an index per sample."""
    order = np.argsort(energy, kind="stable")
    ranks = np.empty(len(energy), dtype=np.int64)
    ranks[order] = np.arange(len(energy))
    return (ranks * n_deciles) // len(energy)


def fragility(samples, mask):
    """
    Fraction of adjacent differences inside `mask` with |d| <= 1.

    This is the structural quantity the argument turns on: the share of
    neighbouring pairs where a one-step perturbation can flip a comparison.
    """
    d = np.abs(np.diff(samples.astype(np.int64)))
    m = mask[:-1] & mask[1:]
    if not m.any():
        return float("nan"), float("nan")
    return float(np.mean(d[m] <= 1)), float(np.mean(d[m]))


def measure_file(path, rng, flips=FLIPS_PER_DECILE):
    """Per-decile detector perturbation for one cover file."""
    cover = read_mono(path).astype(np.int64)
    if len(cover) < FRAME * N_DECILES * 4:
        return []

    energy = sample_energy(cover)
    deciles = decile_assignment(energy)

    base_rs = rs_analysis(samples=cover)["rs_diff"]
    base_spa = abs(sample_pair_analysis(samples=cover)["beta"])

    rows = []
    for d in range(N_DECILES):
        idx = np.flatnonzero(deciles == d)
        if len(idx) < flips:
            continue

        mask = np.zeros(len(cover), dtype=bool)
        mask[idx] = True
        frag, mean_delta = fragility(cover, mask)

        chosen = rng.choice(idx, size=flips, replace=False)
        stego = cover.copy()
        stego[chosen] ^= 1                      # LSB substitution, this decile only

        rs = rs_analysis(samples=stego)["rs_diff"]
        spa = abs(sample_pair_analysis(samples=stego)["beta"])

        rows.append({
            "file": os.path.basename(path),
            "decile": d,
            "mean_energy": float(np.mean(energy[idx])),
            "fragile_fraction": frag,
            "mean_abs_diff": mean_delta,
            "d_rs_per_bit": (rs - base_rs) / flips,
            "d_spa_per_bit": (spa - base_spa) / flips,
            "flips": flips,
        })
    return rows


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    say("=" * 78)
    say("  WHY ENERGY-WEIGHTED PLACEMENT REDUCES DETECTABILITY")
    say("=" * 78)
    say(f"  Fixed {FLIPS_PER_DECILE} LSB flips confined to one energy decile at a time,")
    say(f"  {N_DECILES} equal-population deciles, {FRAME}-sample frames.")
    say("  Decile 0 = quietest, decile 9 = loudest.")
    say()

    for domain in DOMAINS:
        d_path = os.path.join("dataset", domain)
        if not os.path.isdir(d_path):
            continue
        for name in sorted(os.listdir(d_path)):
            if not name.endswith(".wav"):
                continue
            try:
                got = measure_file(os.path.join(d_path, name), rng)
            except (UnreadableAudio, ValueError) as exc:
                say(f"  [SKIP] {name}: {exc}")
                continue
            if got:
                for r in got:
                    r["domain"] = domain
                rows.extend(got)
                print(f"  measured {domain}/{name}")

    if not rows:
        say("  No files large enough to measure.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---- aggregate ----
    say()
    say(f"  {'decile':>7}{'mean energy':>14}{'|d|<=1':>10}{'mean |d|':>11}"
        f"{'d(rs_diff)/bit':>17}{'d(spa)/bit':>14}")
    say("  " + "-" * 74)

    per_decile = {}
    for d in range(N_DECILES):
        sub = [r for r in rows if r["decile"] == d]
        if not sub:
            continue
        rec = {
            "energy": np.mean([r["mean_energy"] for r in sub]),
            "fragile": np.nanmean([r["fragile_fraction"] for r in sub]),
            "absdiff": np.nanmean([r["mean_abs_diff"] for r in sub]),
            "rs": np.mean([r["d_rs_per_bit"] for r in sub]),
            "spa": np.mean([r["d_spa_per_bit"] for r in sub]),
            "n": len(sub),
        }
        per_decile[d] = rec
        say(f"  {d:>7}{rec['energy']:14.1f}{100*rec['fragile']:9.1f}%"
            f"{rec['absdiff']:11.1f}{rec['rs']:17.3e}{rec['spa']:14.3e}")

    say()
    lo, hi = per_decile.get(0), per_decile.get(N_DECILES - 1)
    if lo and hi:
        say("  QUIETEST vs LOUDEST decile, per embedded bit:")
        for key, label in (("rs", "RS  rs_diff"), ("spa", "SPA |beta|")):
            ratio = abs(lo[key]) / abs(hi[key]) if hi[key] else float("inf")
            say(f"    {label}: {abs(lo[key]):.3e} -> {abs(hi[key]):.3e}"
                f"   ({ratio:,.1f}x less perturbation in the loudest decile)")
        say(f"    fragile pairs (|d|<=1): {100*lo['fragile']:.1f}% -> "
            f"{100*hi['fragile']:.1f}%")
        say(f"    mean |adjacent difference|: {lo['absdiff']:.1f} -> {hi['absdiff']:.1f}")

    # Rank correlation between energy and perturbation — the quantitative form
    # of the prediction.
    from scipy.stats import spearmanr
    ds = sorted(per_decile)
    for key, label in (("rs", "RS"), ("spa", "SPA")):
        rho, p = spearmanr([per_decile[d]["energy"] for d in ds],
                           [abs(per_decile[d][key]) for d in ds])
        say(f"\n  Spearman(energy, |perturbation per bit|) for {label}: "
            f"rho = {rho:+.3f}, p = {p:.4f}")

    say()
    say("  A negative rho confirms the prediction: the same number of embedded")
    say("  bits perturbs the detectors less when placed in higher-energy frames.")
    say("  This is the mechanism the RMS weighting exploits, and it accounts for")
    say("  the ablation result (+25.0 pp for energy weighting, -11.4 pp for")
    say("  randomisation): randomisation relocates bits without regard to")
    say("  fragility, whereas energy weighting moves them away from it.")

    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(_out) + "\n")
    print(f"\n[SAVED] {OUT_CSV}")
    print(f"[SAVED] {REPORT}")


if __name__ == "__main__":
    main()
