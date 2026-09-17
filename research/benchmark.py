"""
benchmark.py — Corrected benchmark sweep.

Differences from the original research_benchmark.py, all of which change results:

  1. Three genuinely distinct embedding strategies. The original called
     encode_message() for both the "Sequential" and the "Randomized" arm, so two
     of its three columns were the same algorithm and differed only by the random
     AES IV. Arms now dispatch through core.embedding, where the strategy is a
     parameter rather than a branch.
  2. A clean-cover control at 0% payload. Without it there is no false-positive
     baseline and no way to tell whether a detector responds to the embedding or
     to the audio.
  3. A 2x3 ablation, {plaintext, AES} x {sequential, uniform, adaptive}, which
     isolates the contribution of the crypto layer, the randomisation and the
     energy weighting separately.
  4. Extraction is verified on every run and the bit error rate recorded. The
     original measured eight metrics, none of which checked that the payload
     came back.
  5. Encode and decode wall-clock time recorded, for the complexity analysis.
  6. Raw detector statistics stored alongside the verdicts, so ROC curves and
     significance tests can be computed afterwards without re-running.
  7. Failures are logged, not swallowed by a bare `except: continue`.

Run:  python research/calibrate.py      (once, to derive thresholds)
      python research/benchmark.py
"""

import csv
import os
import sys
import time
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio_io import UnreadableAudio, read_mono
from core.embedding import embed, extract
from core.metrics import (
    bit_error_rate,
    calculate_psnr,
    calculate_snr,
    lsb_entropy,
    spectral_flatness,
)
from core.steganalysis import STAT_KEYS, load_thresholds, run_full_steganalysis

PASSWORD = "research_password_123"
DOMAINS = ["ambient", "instrumental", "short_voice", "speech"]
PAYLOAD_RATES = [0.01, 0.05, 0.10, 0.25, 0.50]
ABLATION_RATES = [0.10, 0.50]
LSB_BITS = 1

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_CSV = os.path.join(HERE, "results_main.csv")
ABLATION_CSV = os.path.join(HERE, "results_ablation.csv")
FAILURE_LOG = os.path.join(HERE, "benchmark_failures.log")

DETECTOR_COLUMNS = [f"stat_{name}" for name in STAT_KEYS] + \
                   [f"det_{name}" for name in STAT_KEYS]

MAIN_HEADERS = [
    "Domain", "Filename", "Method", "Payload_Rate", "LSB_Bits",
    "Message_Bytes", "PSNR", "SNR", "SF_Cover", "SF_Stego", "SF_Change",
    "Entropy_Cover", "Entropy_Stego", "Entropy_Change",
    "Extraction_OK", "BER", "Encode_Seconds", "Decode_Seconds",
] + DETECTOR_COLUMNS + ["Detected_Count", "Discriminative_Detected", "Verdict"]

ABLATION_HEADERS = [
    "Domain", "Filename", "Strategy", "Encrypted", "Payload_Rate",
    "PSNR", "Extraction_OK", "BER",
] + DETECTOR_COLUMNS + ["Detected_Count", "Discriminative_Detected", "Verdict"]


def log_failure(context, exc):
    with open(FAILURE_LOG, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== {context} ===\n{traceback.format_exc()}\n")
    print(f"    [FAIL] {context}: {type(exc).__name__}: {exc}")


def list_cover_files(dataset_root="dataset"):
    """Every readable cover, with its capacity. Unreadable files are reported."""
    covers = []
    for domain in DOMAINS:
        path = os.path.join(dataset_root, domain)
        if not os.path.isdir(path):
            continue
        for filename in sorted(os.listdir(path)):
            if not filename.endswith(".wav"):
                continue
            full = os.path.join(path, filename)
            try:
                samples = read_mono(full)
            except (UnreadableAudio, ValueError) as exc:
                print(f"  [SKIP] {domain}/{filename}: {exc}")
                continue
            covers.append({
                "domain": domain, "filename": filename, "path": full,
                "capacity_bytes": (len(samples) * LSB_BITS) // 8,
            })
    return covers


def verdict_for(discriminative_hits):
    if discriminative_hits == 0:
        return "SAFE"
    if discriminative_hits == 1:
        return "SUSPICIOUS"
    return "DETECTED"


def detector_columns(battery):
    row = {}
    for name in STAT_KEYS:
        row[f"stat_{name}"] = battery[name][STAT_KEYS[name]]
        row[f"det_{name}"] = int(battery[name]["detected"])
    return row


def load_done(path, key_fields):
    """Resume support: which rows already exist."""
    done = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                done.add(tuple(row[f] for f in key_fields))
    return done


def open_appending(path, headers):
    fresh = not os.path.exists(path)
    handle = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=headers)
    if fresh:
        writer.writeheader()
    return handle, writer


# -----------------------------------------------------------------------------
#  CONTROL + MAIN SWEEP
# -----------------------------------------------------------------------------

def run_main_sweep(covers, thresholds):
    key_fields = ["Domain", "Filename", "Method", "Payload_Rate"]
    done = load_done(MAIN_CSV, key_fields)
    handle, writer = open_appending(MAIN_CSV, MAIN_HEADERS)

    try:
        for cover in covers:
            # Read each WAV exactly once per run and share the array across all
            # metrics. Passing paths made every metric re-decode the file, which
            # dominated the runtime.
            cover_samples = read_mono(cover["path"])
            sf_cover = spectral_flatness(cover_samples)
            ent_cover = lsb_entropy(cover_samples)

            # --- clean-cover control: the 0% payload row ---
            control_key = (cover["domain"], cover["filename"], "Control", "0%")
            if control_key not in done:
                try:
                    battery = run_full_steganalysis(
                        cover["path"], samples=cover_samples, thresholds=thresholds
                    )
                    summary = battery["detection_summary"]
                    row = {
                        "Domain": cover["domain"], "Filename": cover["filename"],
                        "Method": "Control", "Payload_Rate": "0%",
                        "LSB_Bits": LSB_BITS, "Message_Bytes": 0,
                        "PSNR": "", "SNR": "",
                        "SF_Cover": sf_cover, "SF_Stego": sf_cover, "SF_Change": 0.0,
                        "Entropy_Cover": ent_cover, "Entropy_Stego": ent_cover,
                        "Entropy_Change": 0.0,
                        "Extraction_OK": "", "BER": "",
                        "Encode_Seconds": "", "Decode_Seconds": "",
                        "Detected_Count": summary["methods_triggered"],
                        "Discriminative_Detected": summary["discriminative_triggered"],
                        "Verdict": verdict_for(summary["discriminative_triggered"]),
                    }
                    row.update(detector_columns(battery))
                    writer.writerow(row)
                    handle.flush()
                    print(f"  {cover['domain']}/{cover['filename']} | Control")
                except Exception as exc:
                    log_failure(f"control {cover['filename']}", exc)

            # --- embedded arms ---
            for rate in PAYLOAD_RATES:
                rate_label = f"{int(rate * 100)}%"
                n_bytes = max(10, int(cover["capacity_bytes"] * rate) - 40)
                message = "X" * n_bytes

                for method, strategy in (("Sequential", "sequential"),
                                         ("Randomized", "uniform"),
                                         ("Adaptive", "adaptive")):
                    key = (cover["domain"], cover["filename"], method, rate_label)
                    if key in done:
                        continue

                    stego = os.path.join(HERE, f"_tmp_{method}_{os.getpid()}.wav")
                    print(f"  {cover['domain']}/{cover['filename']} | {method} @ {rate_label}")
                    try:
                        t0 = time.perf_counter()
                        embed(cover["path"], stego, message, PASSWORD,
                              LSB_BITS, strategy, encrypt=True)
                        encode_seconds = time.perf_counter() - t0

                        t0 = time.perf_counter()
                        try:
                            recovered = extract(stego, PASSWORD, LSB_BITS,
                                                strategy, encrypt=True)
                        except Exception:
                            recovered = None
                        decode_seconds = time.perf_counter() - t0

                        stego_samples = read_mono(stego)
                        battery = run_full_steganalysis(
                            stego, samples=stego_samples, thresholds=thresholds
                        )
                        summary = battery["detection_summary"]
                        sf_stego = spectral_flatness(stego_samples)
                        ent_stego = lsb_entropy(stego_samples)

                        row = {
                            "Domain": cover["domain"], "Filename": cover["filename"],
                            "Method": method, "Payload_Rate": rate_label,
                            "LSB_Bits": LSB_BITS, "Message_Bytes": n_bytes,
                            "PSNR": calculate_psnr(cover_samples, stego_samples),
                            "SNR": calculate_snr(cover_samples, stego_samples),
                            "SF_Cover": sf_cover, "SF_Stego": sf_stego,
                            "SF_Change": abs(sf_stego - sf_cover),
                            "Entropy_Cover": ent_cover, "Entropy_Stego": ent_stego,
                            "Entropy_Change": abs(ent_stego - ent_cover),
                            "Extraction_OK": int(recovered == message),
                            "BER": bit_error_rate(message, recovered),
                            "Encode_Seconds": round(encode_seconds, 4),
                            "Decode_Seconds": round(decode_seconds, 4),
                            "Detected_Count": summary["methods_triggered"],
                            "Discriminative_Detected": summary["discriminative_triggered"],
                            "Verdict": verdict_for(summary["discriminative_triggered"]),
                        }
                        row.update(detector_columns(battery))
                        writer.writerow(row)
                        handle.flush()
                    except Exception as exc:
                        log_failure(f"{cover['filename']} {method} {rate_label}", exc)
                    finally:
                        if os.path.exists(stego):
                            os.remove(stego)
    finally:
        handle.close()


# -----------------------------------------------------------------------------
#  ABLATION — 2x3 factorial
# -----------------------------------------------------------------------------

def run_ablation(covers, thresholds):
    key_fields = ["Domain", "Filename", "Strategy", "Encrypted", "Payload_Rate"]
    done = load_done(ABLATION_CSV, key_fields)
    handle, writer = open_appending(ABLATION_CSV, ABLATION_HEADERS)

    try:
        for cover in covers:
            for rate in ABLATION_RATES:
                rate_label = f"{int(rate * 100)}%"
                n_bytes = max(10, int(cover["capacity_bytes"] * rate) - 40)
                message = "X" * n_bytes

                for strategy in ("sequential", "uniform", "adaptive"):
                    for encrypt in (False, True):
                        key = (cover["domain"], cover["filename"], strategy,
                               str(int(encrypt)), rate_label)
                        if key in done:
                            continue

                        stego = os.path.join(HERE, f"_abl_{os.getpid()}.wav")
                        print(f"  {cover['filename']} | {strategy} "
                              f"{'AES' if encrypt else 'plain'} @ {rate_label}")
                        try:
                            embed(cover["path"], stego, message, PASSWORD,
                                  LSB_BITS, strategy, encrypt=encrypt)
                            try:
                                recovered = extract(stego, PASSWORD, LSB_BITS,
                                                    strategy, encrypt=encrypt)
                            except Exception:
                                recovered = None

                            stego_samples = read_mono(stego)
                            battery = run_full_steganalysis(
                                stego, samples=stego_samples, thresholds=thresholds
                            )
                            summary = battery["detection_summary"]

                            row = {
                                "Domain": cover["domain"],
                                "Filename": cover["filename"],
                                "Strategy": strategy,
                                "Encrypted": int(encrypt),
                                "Payload_Rate": rate_label,
                                "PSNR": calculate_psnr(cover["path"], stego_samples),
                                "Extraction_OK": int(recovered == message),
                                "BER": bit_error_rate(message, recovered),
                                "Detected_Count": summary["methods_triggered"],
                                "Discriminative_Detected": summary["discriminative_triggered"],
                                "Verdict": verdict_for(summary["discriminative_triggered"]),
                            }
                            row.update(detector_columns(battery))
                            writer.writerow(row)
                            handle.flush()
                        except Exception as exc:
                            log_failure(
                                f"ablation {cover['filename']} {strategy} "
                                f"{encrypt} {rate_label}", exc
                            )
                        finally:
                            if os.path.exists(stego):
                                os.remove(stego)
    finally:
        handle.close()


def main():
    print("=" * 72)
    print("  COVERTWAVE BENCHMARK (corrected)")
    print("=" * 72)

    thresholds = load_thresholds()
    saturated = [n for n, s in thresholds.items() if s.get("saturated")]
    if saturated:
        print(f"  Saturated (non-discriminative) detectors: {', '.join(saturated)}")

    covers = list_cover_files()
    print(f"  Cover files: {len(covers)}")
    print(f"  Main sweep : {len(covers)} x ({len(PAYLOAD_RATES)} rates x 3 methods + 1 control)"
          f" = {len(covers) * (len(PAYLOAD_RATES) * 3 + 1)} runs")
    print(f"  Ablation   : {len(covers)} x {len(ABLATION_RATES)} rates x 6 arms"
          f" = {len(covers) * len(ABLATION_RATES) * 6} runs\n")

    run_main_sweep(covers, thresholds)
    run_ablation(covers, thresholds)

    print(f"\n[DONE] {MAIN_CSV}")
    print(f"[DONE] {ABLATION_CSV}")


if __name__ == "__main__":
    main()
