#!/usr/bin/env python3
import os
import sys
import time
import re
from pathlib import Path

log_file = '/tmp/train_output.log'
checkpoint_dir = '/home/user/StegaStamp-plus'

critical_steps = {
    500: "First checkpoint test",
    1000: "Step 1000 - verify learning",
    5000: "Step 5000 - mid-phase check",
    10000: "First saved checkpoint",
    50000: "Halfway through training",
    100000: "Final phase",
    140000: "Training complete"
}

def parse_step_output(line):
    """Extract step number and loss values from training output"""
    match = re.search(r'Step (\d+)/140000.*Secret:([\d.]+).*L2:([\d.]+).*Total:([\d.]+)', line)
    if match:
        step = int(match.group(1))
        secret_loss = float(match.group(2))
        l2_loss = float(match.group(3))
        total_loss = float(match.group(4))
        return step, secret_loss, l2_loss, total_loss
    return None

def parse_debug_output(line):
    """Extract debug step output"""
    match = re.search(r'\[DEBUG Step (\d+)\].*Residual magnitude: ([\d.]+), L2: ([\d.]+)', line)
    if match:
        step = int(match.group(1))
        residual_mag = float(match.group(2))
        l2_loss = float(match.group(3))
        return step, residual_mag, l2_loss
    return None

def check_training_health(secret_loss, l2_loss, residual_mag, step):
    """Check if training looks healthy"""
    issues = []

    # Check for loss at random baseline
    if step >= 500 and secret_loss > 0.69:
        issues.append(f"⚠ Secret loss {secret_loss:.4f} near random baseline 0.6931")

    # Check for zero residuals (collapse)
    if step >= 500 and residual_mag < 0.01:
        issues.append(f"⚠ Residual magnitude {residual_mag:.6f} very small - encoder may be collapsing")

    # Check for NaN
    if secret_loss != secret_loss:  # NaN check
        issues.append("✗ Secret loss is NaN!")

    if l2_loss != l2_loss:  # NaN check
        issues.append("✗ L2 loss is NaN!")

    return issues

def main():
    print("=" * 70)
    print("TRAINING MONITOR - Real-time Loss Tracking")
    print("=" * 70)
    print(f"Monitoring: {log_file}")
    print(f"Critical milestones: {list(critical_steps.keys())}")
    print("=" * 70)

    seen_steps = set()
    last_line_pos = 0
    monitored_steps = set()

    while True:
        if not os.path.exists(log_file):
            print("Waiting for log file to appear...")
            time.sleep(5)
            continue

        try:
            with open(log_file, 'r') as f:
                f.seek(last_line_pos)
                lines = f.readlines()
                last_line_pos = f.tell()

                for line in lines:
                    # Parse step output
                    result = parse_step_output(line)
                    if result:
                        step, secret_loss, l2_loss, total_loss = result
                        if step not in seen_steps:
                            seen_steps.add(step)

                            # Check for critical milestones
                            if step in critical_steps:
                                print(f"\n{'='*70}")
                                print(f"CHECKPOINT: Step {step} - {critical_steps[step]}")
                                print(f"{'='*70}")
                                print(f"  Secret loss: {secret_loss:.6f}")
                                print(f"  L2 loss: {l2_loss:.6f}")
                                print(f"  Total loss: {total_loss:.6f}")
                                monitored_steps.add(step)

                                # Estimate training time
                                elapsed_min = (time.time() - start_time) / 60 if 'start_time' in globals() else 0
                                if step > 0 and elapsed_min > 0:
                                    steps_per_min = step / elapsed_min
                                    eta_min = (140000 - step) / steps_per_min
                                    print(f"  ETA: {eta_min:.0f} minutes ({eta_min/60:.1f} hours)")

                            # Check health at critical steps
                            if step in [500, 1000, 5000, 10000]:
                                print(f"\nStep {step}:")
                                print(f"  Secret: {secret_loss:.6f}, L2: {l2_loss:.6f}, Total: {total_loss:.6f}")
                                issues = check_training_health(secret_loss, l2_loss, 0, step)
                                if issues:
                                    for issue in issues:
                                        print(f"  {issue}")

                    # Parse debug output
                    debug_result = parse_debug_output(line)
                    if debug_result:
                        step, residual_mag, l2_loss = debug_result
                        if step not in seen_steps:
                            seen_steps.add(step)
                            if step < 10:
                                print(f"Step {step}: Residual={residual_mag:.6f}, L2={l2_loss:.6f}")

                # Check for checkpoint files
                for checkpoint_step in [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]:
                    if checkpoint_step not in monitored_steps:
                        enc_path = f"{checkpoint_dir}/encoder_100bit_step_{checkpoint_step}.keras"
                        dec_path = f"{checkpoint_dir}/decoder_100bit_step_{checkpoint_step}.keras"

                        if os.path.exists(enc_path) and os.path.exists(dec_path):
                            file_size_mb = os.path.getsize(enc_path) / (1024*1024)
                            print(f"\n✓ Checkpoint created: Step {checkpoint_step} ({file_size_mb:.0f}MB)")
                            monitored_steps.add(checkpoint_step)

        except Exception as e:
            print(f"Error reading log: {e}")

        time.sleep(10)

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)
