<p align="center">
  <img src="frontend/logo.png" alt="CovertWave Logo" width="80">
</p>

<h1 align="center">CovertWave</h1>

<p align="center">
  <strong>Advanced Audio Steganography with Adaptive Energy-Weighted Embedding</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/AES--256--CBC-Encryption-00ff88" alt="AES-256">
</p>

---

CovertWave is a full-stack audio steganography system that hides encrypted messages inside WAV audio files using an **energy-weighted stochastic LSB embedding** algorithm. Embedding stays above **95 dB PSNR** at every payload tested, and biasing bit placement toward high-energy frames raises evasion of classical steganalysis by **25.2 +/- 4.4 percentage points** at 50% capacity over uniform random placement.

This code backs the paper *"Energy-Weighted Stochastic LSB: Enhancing Steganalysis Evasion in Temporal Audio Steganography"*, presented at the **7th IEEE INDISCON 2026** (MNIT Jaipur, 11-13 September 2026).

> **Note on results.** The figures above come from the corrected benchmark, and they
> supersede the numbers in the originally submitted version of the paper. See
> [Corrections to the original study](#corrections-to-the-original-study) below.

## Features

- **Adaptive Energy-Weighted Embedding** — Stochastically distributes payload bits into high-energy audio frames, making detection statistically infeasible
- **AES-256-CBC Encryption** — Messages are encrypted before embedding, providing robust confidentiality
- **Multi-bit LSB Support** — Choose between 1-bit (highest quality), 2-bit (balanced), or 4-bit (high capacity) embedding
- **Real-time Capacity Meter** — Live progress bar with color-coded security warnings (green/yellow/red)
- **Extraction Payload Map** — Dual-axis Chart.js visualization showing where encrypted fragments are hidden in the audio waveform
- **4-Method Steganalysis Battery** — Chi-Square, RS Analysis, Sample Pair Analysis, and Histogram Analysis for detection testing
- **Premium Dark UI** — Glassmorphic interface with smooth native scrolling and micro-animations

## Architecture

```
CovertWave/
|-- core/                    # Core steganography engine
|   |-- adaptive.py          # Energy-weighted stochastic embedding algorithm
|   |-- sequential.py        # Naive sequential LSB (baseline)
|   |-- encoder.py           # Standard randomized LSB encoder
|   |-- decoder.py           # Standard LSB decoder
|   |-- embedding.py         # Unified engine: strategy is a parameter
|   |-- payload.py           # AES payload assembly + bit-level substitution
|   |-- crypto.py            # AES-256-CBC encrypt/decrypt module
|   |-- audio_io.py          # Single robust WAV reader
|   |-- metrics.py           # PSNR, SNR, spectral flatness, entropy, BER
|   |-- steganalysis.py      # 4-method detection battery (calibrated)
|
|-- backend/
|   |-- main.py              # FastAPI server (encode/decode/analyze APIs)
|
|-- requirements.txt         # Python dependencies
|
|-- frontend/
|   |-- index.html           # Single-page app structure
|   |-- style.css            # Dark glassmorphic theme
|   |-- app.js               # UI logic, Chart.js visualizations
|   |-- logo.png             # CovertWave icon
|
|-- research/                # Research & Benchmarking
|   |-- calibrate.py         # Derives detector thresholds from clean covers
|   |-- benchmark.py         # Main sweep + control + ablation
|   |-- analyze.py           # Significance tests, power, ablation report
|   |-- roc_analysis.py      # Threshold-free ROC / AUC / P_E
|   |-- seed_variance.py     # Replication over independent password draws
|   |-- theory_energy.py     # Energy-vs-detectability mechanism
|   |-- figures.py           # Publication figures at 600 dpi
|   |-- make_selector_fig.py # Selector illustration (presentation)
|   |-- *.csv                # Results data
|   |-- *_report.txt         # Human-readable summaries
|
|-- figures/                 # All generated figures
|
|-- tests/
|   |-- test_covertwave.py   # Regression suite
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CovertWave.git
cd CovertWave
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
cd backend
python main.py
```

The app will be available at **http://localhost:8000**

### 4. Use the app

1. **Encode** — Upload a WAV file, type your secret message, set a password, and download the stego file
2. **Decode** — Upload the stego file, enter the password, and retrieve your hidden message with a visual payload map
3. **Analyze** — Run the full steganalysis battery on any WAV file to test for hidden data

## Dataset

The audio dataset (44 WAV files across 4 categories: ambient, instrumental, short_voice, speech) is not included in this repository due to its size (~236 MB). To run the benchmark, place your own 16-bit PCM `.wav` files in:

```
dataset/
|-- ambient/
|-- instrumental/
|-- short_voice/
|-- speech/
```

## Research

This project includes a full reproducible research pipeline for benchmarking audio steganography:

Run in order:

```bash
python research/calibrate.py    # derive detector thresholds from clean covers
python research/benchmark.py    # main sweep + control + ablation
python research/analyze.py      # significance tests and ablation report
python research/roc_analysis.py # threshold-free ROC / AUC / P_E
python research/theory_energy.py# energy-vs-detectability mechanism
python research/seed_variance.py# replication over 10 independent draws
python research/figures.py      # figures at 600 dpi
```

- **`calibrate.py`** — Sets each detector's threshold at the 95th percentile of its
  clean-cover distribution, giving a defined 5% false-positive rate instead of a
  hand-chosen constant.
- **`benchmark.py`** — Three genuinely distinct embedding strategies across four
  audio domains and five payload rates, plus a 0%-payload control and a
  {plaintext, AES} x {sequential, uniform, adaptive} ablation. Extraction is
  verified and the bit error rate recorded on every run.
- **`analyze.py`** — Paired McNemar tests, Wilson confidence intervals, per-detector
  power, and component contributions.
- **`roc_analysis.py`** — Threshold-free evaluation. Recomputes ROC curves, AUC and
  P_E (minimum average decision error) per detector from statistics the benchmark
  already stored, using the 0%-payload control as negatives. No re-run needed.
- **`theory_energy.py`** — Measures why energy weighting works. Flips a fixed number
  of LSBs confined to one local-energy decile at a time and records the resulting
  shift in each detector statistic, establishing that detectability per embedded bit
  falls with local energy.
- **`figures.py`** — Grayscale-safe figures at 600 dpi.

### Reproducing

Results depend on the calibration step, so run `calibrate.py` before
`benchmark.py`. Both are resumable: re-running skips rows already present in the
CSVs. `research/research_results_full.csv` preserves the original pre-correction
dataset for reference; every other CSV is from the corrected run.

## Corrections to the original study

The version of this paper first submitted to INDISCON reported results that an audit of
this codebase found were not reproducible. The code was fixed, every experiment re-run,
and the camera-ready paper revised to match. This section records what was wrong, because
the corrected numbers differ from the submitted ones.

**1. Two of the three benchmark arms were the same algorithm.**
The original `research_benchmark.py` had a three-way branch in which the "Sequential" and
"Randomized" cases both called `encode_message()`. No sequential embedder existed. Every
sequential-vs-randomized conclusion in the submitted paper therefore described a difference
that was only the random AES initialisation vector. Fixed by `core/embedding.py`, which
dispatches through a table of strategies so the arms are data rather than branches, and by
`core/sequential.py`, which implements the missing baseline.

**2. The detector battery was measuring the cover, not the payload.**
Thresholds were hand-chosen constants with no false-positive rate attached. Under them the
clean covers scored 63.6% "survival" while stego audio at 1-10% payload scored 62-65% --
the signature of a detector responding to the audio rather than to the embedding. Fixed by
`research/calibrate.py`, which sets each threshold at the 95th percentile of its clean-cover
distribution, giving every detector a defined 5% false-positive rate. This also established
the clean-cover control arm the original study never had: audio containing nothing survives
the battery only 81.8% of the time, and that is the ceiling every result must be read against.

**3. The adaptive decoder corrupted part of the corpus.**
Embedding weights were computed on the raw signal, so the act of embedding perturbed the
very distribution used to locate the payload. The weighted CDF shifted enough to move a draw
across a boundary, and one displaced position corrupts the message -- silently, without
raising. Fixed by computing weights on the LSB-cleared signal (`core/adaptive.py`), so
encoder and decoder derive identical distributions, and by replacing the position draw with
an Efraimidis-Spirakis exponential race, which is prefix-stable and so lets the decoder draw
the header and the payload consistently.

**Why these survived to submission:** the repository had no tests. Nothing asserted that a
message came back intact, and nothing asserted that two strategies produce different output.
Both are now assertions in `tests/test_covertwave.py`.

**What changed in the results.** The corrected benchmark covers 704 conditions with a
clean-cover control, replicated over ten independent password draws. Energy-adaptive
placement raises evasion by 25.2 +/- 4.4 percentage points at 50% payload and 19.5 +/- 3.4
at 25%, positive in all twenty comparisons. A component ablation attributes the gain to the
energy weighting alone (+25.0 points); uniform randomisation contributes -11.4 and encryption
none. `research/research_results_full.csv` is retained as the pre-correction record.


## Tech Stack

- **Backend**: Python, FastAPI, NumPy, SciPy, PyCryptodome
- **Frontend**: Vanilla JS, CSS, Chart.js
- **Encryption**: AES-256-CBC (Cipher Block Chaining mode)
- **Signal Processing**: Frame-level RMS energy analysis, spectral transparency metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
