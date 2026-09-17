"""Slide figure: how the energy-weighted selector places its bits.

Not a paper figure — built for the INDISCON presentation. Uses the real
core.adaptive code paths so the panels show actual behaviour, not a schematic.

    python research/make_selector_fig.py
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

sys.path.insert(0, ".")
from core.adaptive import (compute_frame_energy, energy_to_sample_weights,
                           get_adaptive_positions, get_uniform_positions)

WAV  = "dataset/short_voice/shortvoice10.wav"
OUT  = "figures/fig_selector.png"
F, ALPHA, NPOS, SEED = 512, 0.05, 20000, 20250912

sr, s = wavfile.read(WAV)
if s.ndim == 2:
    s = s[:, 0]
s = s.astype(np.float64)
t = np.arange(len(s)) / sr

energy = compute_frame_energy(s, F)
ftimes = np.arange(len(energy)) * F / sr
w      = energy_to_sample_weights(s, F, ALPHA)
ada    = np.array(get_adaptive_positions(s, NPOS, SEED, F, ALPHA))
uni    = np.array(get_uniform_positions(len(s), NPOS, SEED))

plt.rcParams.update({"font.size": 15, "axes.labelsize": 15, "axes.titlesize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
    "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(4, 1, figsize=(8.5, 9.4), sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1, 1.05], "hspace": 0.34})

ax[0].plot(t, s / 32768.0, lw=0.4, color="#3b4a6b")
ax[0].set_ylabel("amplitude")
ax[0].set_title("(a)  Cover waveform", loc="left", fontweight="bold")

ax[1].step(ftimes, energy / energy.max(), where="post", lw=1.6, color="#c2571a")
ax[1].fill_between(ftimes, 0, energy / energy.max(), step="post",
                   color="#c2571a", alpha=0.18)
ax[1].set_ylabel("RMS (norm.)")
ax[1].set_title(f"(b)  Per-frame RMS energy,  F = {F} samples "
                f"(~{1000*F/sr:.0f} ms)", loc="left", fontweight="bold")

pm = w / w.max()
ax[2].plot(t, pm, lw=1.0, color="#1f6f4a")
ax[2].fill_between(t, 0, pm, color="#1f6f4a", alpha=0.15)
ax[2].axhline(ALPHA, ls="--", lw=1.6, color="#8a1c1c")
ax[2].text(t[-1] * 0.995, ALPHA + 0.05, r"$\alpha$ floor = 0.05",
           ha="right", va="bottom", fontsize=14, color="#8a1c1c")
ax[2].set_ylim(0, 1.15)
ax[2].set_ylabel("p (norm.)")
ax[2].set_title(r"(c)  Selection probability $p_j$", loc="left", fontweight="bold")

bins = np.arange(0, t[-1] + 0.1, 0.1)
ha, _ = np.histogram(ada / sr, bins=bins)
hu, _ = np.histogram(uni / sr, bins=bins)
ctr = (bins[:-1] + bins[1:]) / 2
ax[3].fill_between(ctr, 0, ha, step="mid", color="#1f6f4a", alpha=0.30)
ax[3].step(ctr, ha, where="mid", lw=2.0, color="#1f6f4a", label="energy-adaptive")
ax[3].step(ctr, hu, where="mid", lw=2.0, color="#6e6e6e", ls="--",
           label="uniform random")
ax[3].set_ylabel("bits / 100 ms")
ax[3].set_xlabel("time (s)")
ax[3].set_title(f"(d)  Where {NPOS:,} embedded bits actually land",
                loc="left", fontweight="bold")
ax[3].legend(loc="upper right", frameon=False)

for a in ax:
    a.set_xlim(0, t[-1])
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")

loud = energy > 0.10 * energy.max()
fi = lambda p: np.clip(p // F, 0, len(energy) - 1)
print(f"wrote {OUT}")
print(f"loud frames = {loud.mean()*100:.0f}% of duration")
print(f"adaptive: {loud[fi(ada)].mean()*100:.0f}% of bits in loud frames")
print(f"uniform : {loud[fi(uni)].mean()*100:.0f}% of bits in loud frames")
print(f"mean weight ratio adaptive/uniform = {w[ada].mean()/w[uni].mean():.2f}x")
