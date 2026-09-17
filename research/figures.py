"""
figures.py — Publication figures from the corrected benchmark.

Changes that matter for the camera-ready:

  * 600 dpi, up from 300 (reviewers asked for higher resolution).
  * Grayscale-safe: every series carries a distinct marker AND line style, so
    identity survives black-and-white printing and colour-vision deficiency.
    Colour alone never encodes a series.
  * Survival curves carry 95% Wilson confidence intervals, and the clean-cover
    baseline is drawn as a reference line — without it a survival number cannot
    be interpreted, because a detector that fires on clean audio produces
    "survival" figures that say nothing about the embedding.
  * A new detector-power figure showing false-positive rate against
    true-positive rate per detector. Two of the four classical detectors have no
    power on 16-bit audio and the figure says so.
  * A new ablation figure isolating the contribution of each component.
  * Single y-axis everywhere.

Run:  python research/figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.steganalysis import STAT_KEYS, load_thresholds

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_CSV = os.path.join(HERE, "results_main.csv")
ABLATION_CSV = os.path.join(HERE, "results_ablation.csv")
OUT_DIR = os.path.join(HERE, os.pardir, "figures")

DPI = 600

# Categorical slots 1-3 of the validated palette; these three clear the
# all-pairs CVD and normal-vision separation floors. Marker and line style
# repeat the identity so colour is never the only channel.
COLORS = {"Sequential": "#2a78d6", "Randomized": "#eb6834", "Adaptive": "#1baf7a"}
MARKERS = {"Sequential": "o", "Randomized": "s", "Adaptive": "D"}
LINESTYLES = {"Sequential": "-", "Randomized": "--", "Adaptive": "-."}

METHODS = ["Sequential", "Randomized", "Adaptive"]
RATES = ["1%", "5%", "10%", "25%", "50%"]
RATE_VALUES = [1, 5, 10, 25, 50]

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d8d4"
NEUTRAL_BAR = "#b8b7b2"
ACCENT_BAR = "#2a78d6"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def wilson(successes, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def tidy(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, linestyle=":", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# A figure destined for a paper carries no explanatory text of its own. The
# document sets the caption in the body font at the correct width; text rendered
# into the PNG duplicates it at the wrong size, cannot be edited or translated,
# and reads as an artefact. Set this True only for standalone previews.
EMBED_CAPTIONS = False


def save(fig, name, caption_lines=None):
    """
    Writes the figure. With EMBED_CAPTIONS off, caption_lines are printed to
    stdout for pasting into the document's own \\caption{} rather than drawn.
    """
    if caption_lines:
        if EMBED_CAPTIONS:
            fig.subplots_adjust(bottom=0.32)
            fig.text(0.02, 0.012, "\n".join(caption_lines),
                     fontsize=8, color=MUTED, ha="left", va="bottom",
                     linespacing=1.5)
        else:
            print("    [caption] " + " ".join(caption_lines))

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# -----------------------------------------------------------------------------

def fig_rate_distortion(df):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    for method in METHODS:
        means = [df[(df.Method == method) & (df.Payload_Rate == r)]["PSNR"].mean()
                 for r in RATES]
        ax.plot(RATE_VALUES, means, color=COLORS[method], marker=MARKERS[method],
                linestyle=LINESTYLES[method], linewidth=1.8, markersize=6,
                label=method, markeredgecolor="white", markeredgewidth=0.8)

    ax.axhline(40, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate("40 dB transparency threshold", xy=(1.5, 43), fontsize=8.5, color=MUTED)

    ax.set_xlabel("Payload rate (% of capacity)")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("")
    ax.set_xticks(RATE_VALUES)
    ax.legend(frameon=False, loc="upper right")
    tidy(ax)

    save(fig, "fig1_rate_distortion.png", [
        "Curves coincide by construction: at a given payload every strategy modifies",
        "the same number of samples, so PSNR cannot distinguish them. Position",
        "strategy is not a rate-distortion variable.",
    ])


def fig_survival(df, control_rate):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for method in METHODS:
        centres, los, his = [], [], []
        for rate in RATES:
            sub = df[(df.Method == method) & (df.Payload_Rate == rate)]
            s, n = int(sub["Survived"].sum()), len(sub)
            lo, hi = wilson(s, n)
            centres.append(100 * s / n if n else np.nan)
            los.append(100 * lo)
            his.append(100 * hi)

        ax.fill_between(RATE_VALUES, los, his, color=COLORS[method], alpha=0.09,
                        linewidth=0, zorder=1)
        ax.plot(RATE_VALUES, centres, color=COLORS[method], marker=MARKERS[method],
                linestyle=LINESTYLES[method], linewidth=1.8, markersize=6,
                label=method, markeredgecolor="white", markeredgewidth=0.8, zorder=3)

    ax.axhline(control_rate, color=INK, linestyle=(0, (4, 2)), linewidth=1.3, zorder=2)
    ax.annotate(f"clean-cover baseline ({control_rate:.0f}%)",
                xy=(24, control_rate + 2.0), fontsize=8.5, color=INK)

    ax.set_xlabel("Payload rate (% of capacity)")
    ax.set_ylabel("Files evading detection (%)")
    ax.set_title("")
    ax.set_xticks(RATE_VALUES)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower left")
    tidy(ax)

    save(fig, "fig2_security_survival.png", [
        "The baseline is what the battery reports for cover audio containing nothing.",
        "Survival below it is the only evidence of detection; a curve sitting at the",
        "baseline means the detector is responding to the audio, not to the payload.",
    ])


def fig_detector_power(df):
    control = df[df.Method == "Control"]
    high = df[(df.Method != "Control") & (df.Payload_Rate == "50%")]
    thresholds = load_thresholds()

    names = list(STAT_KEYS)
    labels = ["Chi-Square", "RS Analysis", "Sample Pair", "Histogram"]
    fpr = [100 * control[f"det_{n}"].mean() for n in names]
    tpr = [100 * high[f"det_{n}"].mean() for n in names]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - width / 2 - 0.01, fpr, width, label="False positives (clean covers)",
           color=NEUTRAL_BAR, edgecolor="white", linewidth=1.0, zorder=3)
    ax.bar(x + width / 2 + 0.01, tpr, width, label="True positives (50% payload)",
           color=ACCENT_BAR, edgecolor="white", linewidth=1.0, zorder=3)

    ceiling = max(max(tpr), max(fpr), 1.0)
    saturated = [thresholds.get(n, {}).get("saturated", False) for n in names]

    for xi, (f, t) in enumerate(zip(fpr, tpr)):
        # A zero-height bar is indistinguishable from a missing one, so a
        # saturated detector gets a visible stub on the baseline and explicit
        # "0" labels: the value was measured, and it is zero rather than absent.
        if saturated[xi]:
            for off, col in ((-width / 2 - 0.01, NEUTRAL_BAR),
                             (width / 2 + 0.01, ACCENT_BAR)):
                ax.plot([xi + off - width / 2, xi + off + width / 2], [0, 0],
                        color=col, linewidth=2.8, solid_capstyle="butt",
                        zorder=5, clip_on=False)
            ax.text(xi - width / 2 - 0.01, ceiling * 0.03, "0", ha="center",
                    fontsize=8.5, color=MUTED)
            ax.text(xi + width / 2 + 0.01, ceiling * 0.03, "0", ha="center",
                    fontsize=8.5, color=INK)
            ax.text(xi, ceiling * 0.14, "saturated\n(no power)", ha="center",
                    fontsize=8, color=MUTED, style="italic")
            continue
        ax.text(xi - width / 2 - 0.01, f + ceiling * 0.03, f"{f:.0f}", ha="center",
                fontsize=8.5, color=MUTED)
        ax.text(xi + width / 2 + 0.01, t + ceiling * 0.03, f"{t:.0f}", ha="center",
                fontsize=8.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("", pad=22)
    ax.set_ylim(0, ceiling * 1.15)
    # Legend above the plotting area, where it cannot sit on top of a bar.
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, borderaxespad=0)
    tidy(ax, grid_axis="y")

    save(fig, "fig3_detector_power.png", [
        "A detector whose true-positive bar does not clear its false-positive bar is",
        "not detecting anything. Thresholds are the 95th percentile of each statistic",
        "over the clean covers.",
    ])


def fig_spectral(df):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method in METHODS:
        means = [df[(df.Method == method) & (df.Payload_Rate == r)]["SF_Change"].mean()
                 for r in RATES]
        ax.plot(RATE_VALUES, means, color=COLORS[method], marker=MARKERS[method],
                linestyle=LINESTYLES[method], linewidth=1.8, markersize=6,
                label=method, markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xlabel("Payload rate (% of capacity)")
    ax.set_ylabel("Mean spectral flatness deviation")
    ax.set_title("Spectral transparency (lower is better)", loc="left")
    ax.set_xticks(RATE_VALUES)
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)

    save(fig, "fig4_spectral_transparency.png", [
        "Differences between strategies sit in the sixth decimal place, within run-to-",
        "run noise. Spectral flatness does not separate these methods.",
    ])


def fig_domain(df):
    mid = df[df.Payload_Rate == "25%"]
    domains = sorted(mid["Domain"].unique())

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    width = 0.24
    for i, method in enumerate(METHODS):
        offset = (i - 1) * (width + 0.02)
        data = [mid[(mid.Domain == d) & (mid.Method == method)]["PSNR"].dropna().values
                for d in domains]
        if not any(len(d) for d in data):
            continue
        positions = np.arange(len(domains)) + offset
        bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.2),
                        flierprops=dict(marker=MARKERS[method], markersize=3.5,
                                        markerfacecolor=COLORS[method],
                                        markeredgecolor="none", alpha=0.7))
        for box in bp["boxes"]:
            box.set(facecolor=COLORS[method], alpha=0.55,
                    edgecolor=COLORS[method], linewidth=1.0)
        for element in ("whiskers", "caps"):
            for item in bp[element]:
                item.set(color=COLORS[method], linewidth=1.0)
        ax.plot([], [], color=COLORS[method], linewidth=6, alpha=0.55, label=method)

    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels([d.replace("_", " ") for d in domains])
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Domain-specific fidelity at 25% payload", loc="left")
    ax.legend(frameon=False, loc="lower right", ncol=3)
    tidy(ax, grid_axis="y")
    save(fig, "fig5_domain_robustness.png")


def fig_ablation():
    if not os.path.exists(ABLATION_CSV):
        print("  (no ablation results yet)")
        return
    ab = pd.read_csv(ABLATION_CSV)
    if ab.empty:
        print("  (ablation results empty)")
        return
    ab["Survived"] = (ab["Discriminative_Detected"] == 0).astype(int)

    strategies = ["sequential", "uniform", "adaptive"]
    labels = ["Sequential", "Uniform\nrandom", "Energy-\nadaptive"]
    rates = sorted(ab["Payload_Rate"].unique(), key=lambda r: int(r.rstrip("%")))

    fig, axes = plt.subplots(1, len(rates), figsize=(7.2, 4.0), sharey=True)
    if len(rates) == 1:
        axes = [axes]

    x = np.arange(len(strategies))
    width = 0.36

    for ax, rate in zip(axes, rates):
        sub = ab[ab.Payload_Rate == rate]
        plain = [100 * sub[(sub.Strategy == s) & (sub.Encrypted == 0)]["Survived"].mean()
                 for s in strategies]
        aes = [100 * sub[(sub.Strategy == s) & (sub.Encrypted == 1)]["Survived"].mean()
               for s in strategies]

        ax.bar(x - width / 2 - 0.01, plain, width, label="Plaintext payload",
               color=NEUTRAL_BAR, edgecolor="white", linewidth=1.0, zorder=3)
        ax.bar(x + width / 2 + 0.01, aes, width, label="AES-256 payload",
               color=ACCENT_BAR, edgecolor="white", linewidth=1.0, zorder=3)

        for xi, (p, a) in enumerate(zip(plain, aes)):
            if not np.isnan(p):
                ax.text(xi - width / 2 - 0.01, p + 2, f"{p:.0f}", ha="center",
                        fontsize=8, color=MUTED)
            if not np.isnan(a):
                ax.text(xi + width / 2 + 0.01, a + 2, f"{a:.0f}", ha="center",
                        fontsize=8, color=INK)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"({'ab'[list(rates).index(rate)]}) {rate} payload",
                     loc="left", fontsize=10)
        ax.set_ylim(0, 106)
        tidy(ax, grid_axis="y")

    axes[0].set_ylabel("Files evading detection (%)")
    # Figure-level legend above both panels. Inside an axes it covered the first
    # bar; anchored to one axes it collided with that panel's title.
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, ncol=2, fontsize=9,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))

    save(fig, "fig6_ablation.png", [
        "Encryption changes the payload's statistics, not its placement; the position",
        "strategy is what moves detectability. Comparing the two bars within each",
        "group isolates the contribution of the crypto layer.",
    ])


def fig_energy_theory():
    """Detector perturbation per embedded bit, against local energy decile."""
    path = os.path.join(HERE, "theory_energy.csv")
    if not os.path.exists(path):
        print("  (no theory_energy.csv yet)")
        return
    t = pd.read_csv(path)
    agg = t.groupby("decile").agg(
        energy=("mean_energy", "mean"), fragile=("fragile_fraction", "mean"),
        rs=("d_rs_per_bit", "mean"), spa=("d_spa_per_bit", "mean")).reset_index()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.9))
    fig.subplots_adjust(wspace=0.42)

    ax.plot(agg.decile, np.abs(agg.rs), color=COLORS["Sequential"], marker="o",
            linestyle="-", linewidth=1.8, markersize=6, label="RS analysis",
            markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(agg.decile, np.abs(agg.spa), color=COLORS["Randomized"], marker="s",
            linestyle="--", linewidth=1.8, markersize=6, label="Sample Pair",
            markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Local energy decile (0 = quietest)")
    ax.set_ylabel("|perturbation| per bit")
    ax.set_title("(a)", loc="left", fontsize=10)
    ax.set_xticks(range(10))
    ax.legend(frameon=False)
    tidy(ax)

    ax2.plot(agg.decile, 100 * agg.fragile, color=COLORS["Adaptive"], marker="D",
             linestyle="-.", linewidth=1.8, markersize=6,
             markeredgecolor="white", markeredgewidth=0.8)
    ax2.set_xlabel("Local energy decile (0 = quietest)")
    ax2.set_ylabel("Fragile adjacent pairs (%)")
    ax2.set_title("(b)", loc="left", fontsize=10)
    ax2.set_xticks(range(10))
    tidy(ax2)

    save(fig, "fig7_energy_theory.png", [
        "Left: shift in each detector statistic caused by a fixed number of LSB flips",
        "confined to one energy decile. Right: the structural reason — an LSB flip can",
        "only change a comparison where adjacent samples already differ by 0 or 1, and",
        "such pairs become rare as energy rises. This is the mechanism RMS weighting uses.",
    ])


def fig_roc_pe():
    """P_E by payload for the two detectors that have discriminative power."""
    path = os.path.join(HERE, "roc_results.csv")
    if not os.path.exists(path):
        print("  (no roc_results.csv yet)")
        return
    r = pd.read_csv(path)
    live = [("rs_analysis", "RS Analysis"), ("spa", "Sample Pair")]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
    for ax, (det, label) in zip(axes, live):
        sub = r[r.detector == det]
        for method in METHODS:
            m = sub[sub.method == method].set_index("payload").reindex(RATES)
            ax.plot(RATE_VALUES, m["p_e"], color=COLORS[method], marker=MARKERS[method],
                    linestyle=LINESTYLES[method], linewidth=1.8, markersize=6,
                    label=method, markeredgecolor="white", markeredgewidth=0.8)
        ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.2)
        ax.set_xlabel("Payload rate (% of capacity)")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xticks(RATE_VALUES)
        ax.set_ylim(0, 0.56)
        tidy(ax)

    axes[0].set_ylabel("$P_E$  (higher = better hidden)")
    axes[0].annotate("undetectable", xy=(2, 0.51), fontsize=8.5, color=MUTED)
    axes[0].legend(frameon=False, loc="lower left")

    save(fig, "fig8_pe_by_payload.png", [
        "$P_E$ is the minimum average decision error, the conventional threshold-free",
        "steganalysis measure: 0.5 means the detector is guessing, 0 means it always",
        "catches the payload. Computed against the 0%-payload control on the same",
        "recordings, so no calibrated threshold enters the comparison.",
    ])


def main():
    if not os.path.exists(MAIN_CSV):
        print(f"[ERROR] {MAIN_CSV} not found. Run research/benchmark.py first.")
        return

    df = pd.read_csv(MAIN_CSV)
    df["Survived"] = (df["Discriminative_Detected"] == 0).astype(int)

    control = df[df.Method == "Control"]
    control_rate = 100 * control["Survived"].mean() if len(control) else float("nan")
    embedded = df[df.Method != "Control"]

    print(f"Figures at {DPI} dpi -> {OUT_DIR}")
    fig_rate_distortion(embedded)
    fig_survival(embedded, control_rate)
    fig_detector_power(df)
    fig_spectral(embedded)
    fig_domain(embedded)
    fig_ablation()
    fig_energy_theory()
    fig_roc_pe()


if __name__ == "__main__":
    main()
