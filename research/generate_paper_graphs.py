"""
generate_paper_graphs.py — Final Research Visualization
Generates 5 High-Impact Graphs for IEEE Submission.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "grid.alpha": 0.3
})

def generate_graphs(csv_path="research_results_full.csv", output_dir="graphs_final"):
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['Rate_Val'] = df['Payload_Rate'].str.replace('%', '').astype(int)
    df = df.sort_values('Rate_Val')

    # 1. Rate-Distortion Curve (PSNR vs Payload)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Rate_Val", y="PSNR", hue="Method", style="Method", 
                 markers=True, dashes=False, linewidth=2.5, palette="viridis")
    plt.title("Rate-Distortion Analysis: PSNR vs. Payload Capacity")
    plt.xlabel("Payload Rate (% of Max Capacity)")
    plt.ylabel("Avg PSNR (dB)")
    plt.axhline(y=40, color='r', linestyle='--', label='Transparency Threshold (40dB)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_rate_distortion.png", dpi=300)

    # 2. Security Survival Curve (% files NOT 'DETECTED')
    plt.figure(figsize=(10, 6))
    df['Survived'] = (df['Verdict'] != "DETECTED").astype(int) * 100
    survival = df.groupby(['Rate_Val', 'Method'])['Survived'].mean().reset_index()
    sns.lineplot(data=survival, x="Rate_Val", y="Survived", hue="Method", 
                 markers=True, linewidth=2.5, palette="magma")
    plt.title("Security Survival Curve: Evasion Probability vs. Payload")
    plt.xlabel("Payload Rate (%)")
    plt.ylabel("Files Eving Detection (%)")
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_security_survival.png", dpi=300)

    # 3. Spectral Transparency (SF Change)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Rate_Val", y="SF_Change", hue="Method", palette="Set2")
    plt.title("Spectral Transparency: Distortion of Spectral Flatness")
    plt.xlabel("Payload Rate (%)")
    plt.ylabel("Mean SF Deviation (lower is better)")
    plt.yscale('log') # Use log scale because changes are tiny but significant
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_spectral_transparency.png", dpi=300)

    # 4. Steganalysis Breakdown (methods triggered)
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x="Rate_Val", y="Steg_Detected_Count", hue="Method", 
                  markers=["o", "s", "D"], linestyles=["-", "--", "-."], palette="coolwarm")
    plt.title("Detection Sensitivity Battery Analysis")
    plt.xlabel("Payload Rate (%)")
    plt.ylabel("Avg Detection Triggers")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_detection_battery.png", dpi=300)

    # 5. Domain-Specific Robustness (at 25% payload)
    plt.figure(figsize=(10, 6))
    mid_df = df[df['Rate_Val'] == 25]
    sns.boxplot(data=mid_df, x="Domain", y="PSNR", hue="Method", palette="viridis")
    plt.title("Domain-Specific Robustness at 25% Payload")
    plt.ylabel("PSNR (dB)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/5_domain_robustness.png", dpi=300)

    print(f"[SUCCESS] All 5 IEEE-Impact graphs saved to: {output_dir}")

if __name__ == "__main__":
    generate_graphs()
