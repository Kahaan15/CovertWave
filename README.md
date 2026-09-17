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
CSVs.

## Tech Stack

- **Backend**: Python, FastAPI, NumPy, SciPy, PyCryptodome
- **Frontend**: Vanilla JS, CSS, Chart.js
- **Encryption**: AES-256-CBC (Cipher Block Chaining mode)
- **Signal Processing**: Frame-level RMS energy analysis, spectral transparency metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
