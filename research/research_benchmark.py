"""
research_benchmark.py — Final Research Version
Full Payload Sweep + Complete Metric Battery (PSNR, SNR, SF, Entropy, Steganalysis).
"""

import os
import csv
import time
import numpy as np
from scipy.io import wavfile
from scipy.stats import entropy

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core project modules
from core.encoder import encode_message
from core.adaptive import encode_adaptive
from core.metrics import calculate_psnr, calculate_mse
from core.steganalysis_extended import run_full_steganalysis, _read_samples

# ─────────────────────────────────────────────────────────────────────────────
#  ADVANCED METRICS
# ─────────────────────────────────────────────────────────────────────────────

def get_capacity_bytes(wav_path, lsb_bits=1):
    samples = _read_samples(wav_path)
    return (len(samples) * lsb_bits) // 8

def calculate_snr(orig_wav, stego_wav):
    try:
        orig = _read_samples(orig_wav).astype(np.float64)
        stego = _read_samples(stego_wav).astype(np.float64)
        noise = orig - stego
        sig_p = np.mean(orig ** 2)
        noi_p = np.mean(noise ** 2)
        if noi_p == 0: return 100.0
        return round(float(10 * np.log10(sig_p / noi_p)), 4)
    except: return 0.0

def calculate_spectral_flatness(wav_path):
    try:
        samples = _read_samples(wav_path)
        fft_vals = np.abs(np.fft.rfft(samples))
        fft_vals = fft_vals[fft_vals > 0]
        geom_mean = np.exp(np.mean(np.log(fft_vals)))
        arith_mean = np.mean(fft_vals)
        return round(float(geom_mean / arith_mean), 6)
    except: return 0.0

def calculate_lsb_entropy(wav_path):
    try:
        samples = _read_samples(wav_path)
        lsbs = samples & 1
        counts = np.bincount(lsbs, minlength=2)
        probs = counts / len(lsbs)
        return round(float(entropy(probs, base=2)), 6)
    except: return 0.0

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_grand_sweep(dataset_root="dataset", output_csv="research_results_full.csv"):
    print("\n" + "="*70)
    print("  COVERTWAVE: FINAL RESEARCH BENCHMARK (V3)")
    print("="*70)
    
    password = "research_password_123"
    payload_rates = [0.01, 0.05, 0.10, 0.25, 0.50]
    
    headers = [
        "Domain", "Filename", "Method", "Payload_Rate", 
        "PSNR", "SNR", "SF_Change", "Entropy_Change",
        "Steg_Detected_Count", "Verdict"
    ]
    
    existing_runs = set()
    if os.path.exists(output_csv):
        with open(output_csv, mode="r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    existing_runs.add((row[0], row[1], row[2], row[3]))

    file_mode = "a" if os.path.exists(output_csv) else "w"
    with open(output_csv, mode=file_mode, newline="") as f:
        writer = csv.writer(f)
        if file_mode == "w": writer.writerow(headers)

        domains = ["ambient", "instrumental", "short_voice", "speech"]
        
        for domain in domains:
            domain_path = os.path.join(dataset_root, domain)
            if not os.path.exists(domain_path): continue
            files = [f for f in os.listdir(domain_path) if f.endswith(".wav")]
            
            for filename in files:
                input_wav = os.path.join(domain_path, filename)
                
                try:
                    sf_orig = calculate_spectral_flatness(input_wav)
                    ent_orig = calculate_lsb_entropy(input_wav)
                    max_bytes = get_capacity_bytes(input_wav)
                except: continue

                for rate in payload_rates:
                    num_bytes = int(max_bytes * rate)
                    if num_bytes < 10: num_bytes = 10
                    dummy_message = "X" * num_bytes
                    
                    for method in ["Sequential", "Randomized", "Adaptive"]:
                        rate_str = f"{int(rate*100)}%"
                        if (domain, filename, method, rate_str) in existing_runs:
                            continue
                        
                        print(f"  > {domain}/{filename} | {method} @ {rate_str}")
                        output_wav = "tmp_stego.wav"
                        
                        try:
                            # 1. Embedding
                            if method == "Sequential":
                                encode_message(input_wav, output_wav, dummy_message, password)
                            elif method == "Randomized":
                                encode_message(input_wav, output_wav, dummy_message, password)
                            else:
                                encode_adaptive(input_wav, output_wav, dummy_message, password)
                            
                            # 2. Metrics
                            psnr = calculate_psnr(input_wav, output_wav)
                            snr = calculate_snr(input_wav, output_wav)
                            sf_stego = calculate_spectral_flatness(output_wav)
                            ent_stego = calculate_lsb_entropy(output_wav)
                            
                            sf_change = abs(sf_stego - sf_orig)
                            ent_change = abs(ent_stego - ent_orig)
                            
                            # 3. Steganalysis
                            steg_res = run_full_steganalysis(output_wav)
                            det_count = steg_res["detection_summary"]["methods_triggered"]
                            verdict = "SAFE" if det_count == 0 else ("SUSPICIOUS" if det_count <= 1 else "DETECTED")
                            
                            # 4. Save
                            writer.writerow([
                                domain, filename, method, rate_str, 
                                psnr, snr, sf_change, ent_change,
                                det_count, verdict
                            ])
                            f.flush()
                        except Exception as e:
                            print(f"    [!] Error: {e}")
                        finally:
                            if os.path.exists(output_wav):
                                try: os.remove(output_wav)
                                except: pass

    print(f"\n[SUCCESS] Final Results saved to: {output_csv}")

if __name__ == "__main__":
    run_grand_sweep()
