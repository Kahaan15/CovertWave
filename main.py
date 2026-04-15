import argparse
import os
import sys
from encoder import encode_message, calculate_capacity
from decoder import decode_message
from metrics import (
    calculate_psnr,
    calculate_mse,
    run_psnr_experiment,
    plot_psnr_graph,
    print_capacity_report
)
from steganalysis import (
    chi_square_test,
    compare_files,
    plot_steganalysis_graph,
    print_lsb_distribution
)


# ─────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       Secure Audio Steganography Tool                    ║
║       AES-256 + Password-Seeded Randomized LSB           ║
║       BTech CS Sem VI | Cybersecurity Project 2025-26    ║
╚══════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────
#  COMMAND HANDLERS
# ─────────────────────────────────────────────

def handle_encode(args):
    """
    Handles the 'encode' command.
    Hides a secret message inside a WAV file.
    """
    print(BANNER)
    print("[MODE] ENCODE — Hiding secret message in audio\n")

    # Validate input file
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    # Get message from arg or prompt
    if args.message:
        message = args.message
    else:
        message = input("Enter secret message: ").strip()
        if not message:
            print("[ERROR] Message cannot be empty.")
            sys.exit(1)

    # Get password from arg or prompt
    if args.password:
        password = args.password
    else:
        import getpass
        password = getpass.getpass("Enter password: ").strip()
        if not password:
            print("[ERROR] Password cannot be empty.")
            sys.exit(1)

    # Set output path
    output = args.output if args.output else "stego_output.wav"

    print(f"\n  Input  : {args.input}")
    print(f"  Output : {output}")
    print(f"  LSB    : {args.lsb}-bit mode")
    print()

    try:
        encode_message(
            input_wav  = args.input,
            output_wav = output,
            message    = message,
            password   = password,
            lsb_bits   = args.lsb
        )

        # Auto calculate PSNR after encoding
        psnr = calculate_psnr(args.input, output)
        mse  = calculate_mse(args.input, output)
        print(f"\n[QUALITY] PSNR : {psnr} dB")
        print(f"[QUALITY] MSE  : {mse}")

        if psnr >= 40:
            print("[QUALITY] ✓ Distortion is imperceptible (PSNR ≥ 40 dB)")
        else:
            print("[QUALITY] ⚠ Noticeable distortion (PSNR < 40 dB)")

    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


def handle_decode(args):
    """
    Handles the 'decode' command.
    Extracts and decrypts hidden message from stego WAV file.
    """
    print(BANNER)
    print("[MODE] DECODE — Extracting hidden message from audio\n")

    # Validate input file
    if not os.path.exists(args.input):
        print(f"[ERROR] Stego file not found: {args.input}")
        sys.exit(1)

    # Get password from arg or prompt
    if args.password:
        password = args.password
    else:
        import getpass
        password = getpass.getpass("Enter password: ").strip()
        if not password:
            print("[ERROR] Password cannot be empty.")
            sys.exit(1)

    print(f"  Stego File : {args.input}")
    print(f"  LSB Mode   : {args.lsb}-bit\n")

    try:
        message = decode_message(
            stego_wav = args.input,
            password  = password,
            lsb_bits  = args.lsb
        )
        print(f"\n{'='*50}")
        print(f"  SECRET MESSAGE:")
        print(f"  {message}")
        print(f"{'='*50}\n")

    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


def handle_analyze(args):
    """
    Handles the 'analyze' command.
    Runs steganalysis on a file and optionally compares with original.
    """
    print(BANNER)
    print("[MODE] ANALYZE — Running steganalysis\n")

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    # LSB distribution check
    print_lsb_distribution(args.input)

    # Chi-square test
    print(f"\n[Chi-Square Test] Analyzing: {args.input}")
    result = chi_square_test(args.input)

    print(f"\n  Total Blocks      : {result['total_blocks']}")
    print(f"  Suspicious Blocks : {result['suspicious_blocks']}")
    print(f"  Detection Rate    : {result['detection_rate']}%")
    print(f"  Avg Chi Score     : {result['avg_chi_score']}")
    print(f"  Avg P-Value       : {result['avg_p_value']}")
    print(f"\n  Verdict: {result['verdict']}")

    # If original provided — run full comparison
    if args.original:
        if not os.path.exists(args.original):
            print(f"[ERROR] Original file not found: {args.original}")
            sys.exit(1)

        orig_result, stego_result = compare_files(args.original, args.input)

        if args.graph:
            plot_steganalysis_graph(
                orig_result,
                stego_result,
                save_path=args.graph
            )


def handle_metrics(args):
    """
    Handles the 'metrics' command.
    Runs full PSNR experiment across all LSB modes and plots graph.
    """
    print(BANNER)
    print("[MODE] METRICS — PSNR Experiment across LSB modes\n")

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    # Get message
    if args.message:
        message = args.message
    else:
        message = input("Enter test message for experiment: ").strip()

    # Get password
    if args.password:
        password = args.password
    else:
        import getpass
        password = getpass.getpass("Enter password: ").strip()

    # Run experiment
    results = run_psnr_experiment(args.input, message, password)

    # Plot graph
    graph_path = args.graph if args.graph else "psnr_graph.png"
    plot_psnr_graph(results, save_path=graph_path)


def handle_capacity(args):
    print(BANNER)
    print("[MODE] CAPACITY — Checking audio file capacity\n")

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    print_capacity_report(args.input)
    sys.stdout.flush()


# ─────────────────────────────────────────────
#  ARGUMENT PARSER SETUP
# ─────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog        = "AudioSteg",
        description = "Secure Audio Steganography — AES + Randomized LSB",
        formatter_class = argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── ENCODE ──
    enc = subparsers.add_parser("encode", help="Hide a secret message in a WAV file")
    enc.add_argument("--input",    "-i", required=True,       help="Input WAV file path")
    enc.add_argument("--output",   "-o",                      help="Output stego WAV path (default: stego_output.wav)")
    enc.add_argument("--message",  "-m",                      help="Secret message to hide")
    enc.add_argument("--password", "-p",                      help="Encryption password")
    enc.add_argument("--lsb",      "-l", type=int, default=1,
                     choices=[1, 2, 4],                       help="LSB bits to use: 1, 2, or 4 (default: 1)")

    # ── DECODE ──
    dec = subparsers.add_parser("decode", help="Extract hidden message from stego WAV file")
    dec.add_argument("--input",    "-i", required=True,       help="Stego WAV file path")
    dec.add_argument("--password", "-p",                      help="Decryption password")
    dec.add_argument("--lsb",      "-l", type=int, default=1,
                     choices=[1, 2, 4],                       help="LSB bits used during encoding (default: 1)")

    # ── ANALYZE ──
    ana = subparsers.add_parser("analyze", help="Run steganalysis on a WAV file")
    ana.add_argument("--input",    "-i", required=True,       help="WAV file to analyze")
    ana.add_argument("--original", "-r",                      help="Original WAV for comparison (optional)")
    ana.add_argument("--graph",    "-g",                      help="Save steganalysis graph to this path")

    # ── METRICS ──
    met = subparsers.add_parser("metrics", help="Run PSNR experiment across LSB modes")
    met.add_argument("--input",    "-i", required=True,       help="Input WAV file")
    met.add_argument("--message",  "-m",                      help="Test message for experiment")
    met.add_argument("--password", "-p",                      help="Password for experiment")
    met.add_argument("--graph",    "-g",                      help="Save PSNR graph to this path")

    # ── CAPACITY ──
    cap = subparsers.add_parser("capacity", help="Check how much data a WAV file can hold")
    cap.add_argument("--input",    "-i", required=True,       help="WAV file to check")

    return parser


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    # Route to correct handler
    commands = {
        "encode"   : handle_encode,
        "decode"   : handle_decode,
        "analyze"  : handle_analyze,
        "metrics"  : handle_metrics,
        "capacity" : handle_capacity
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()