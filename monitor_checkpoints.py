#!/usr/bin/env python3
import os
import glob
import json
import subprocess
from datetime import datetime

CHECKPOINT_STEPS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]

print(f"StegaStamp 256-bit Training Monitor")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")

results = {}

for step in CHECKPOINT_STEPS:
    encoder_path = f'encoder_256bit_step_{step}.keras'
    decoder_path = f'decoder_256bit_step_{step}.keras'

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print(f"Step {step:6d}: PENDING (waiting for checkpoint)")
        continue

    print(f"Step {step:6d}: Testing checkpoint...", end=" ", flush=True)

    try:
        result = subprocess.run(
            ['python3', 'test_checkpoint.py', str(step)],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Mean Accuracy:' in line:
                    accuracy_line = line
                elif 'Mean Secret Loss:' in line:
                    loss_line = line
                elif 'Mean L2 Loss:' in line:
                    l2_line = line

            try:
                acc_str = accuracy_line.split(':')[1].strip().split()[0]
                loss_str = loss_line.split(':')[1].strip().split()[0]
                l2_str = l2_line.split(':')[1].strip().split()[0]

                accuracy = float(acc_str)
                secret_loss = float(loss_str)
                l2_loss = float(l2_str)

                results[step] = {
                    'accuracy': accuracy,
                    'secret_loss': secret_loss,
                    'l2_loss': l2_loss
                }

                status = "✓ PASS" if accuracy > 0.5 else "✗ FAIL"
                print(f"{status} | Acc: {accuracy:.4f} | SecLoss: {secret_loss:.4f} | L2Loss: {l2_loss:.4f}")
            except:
                print("ERROR parsing output")
        else:
            print("ERROR running test")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")

print(f"{'='*80}")
print(f"Summary of Results:")
print(f"{'='*80}")
print(f"Step     | Accuracy | SecLoss | L2Loss  | Status")
print(f"{'-'*80}")

for step in CHECKPOINT_STEPS:
    if step in results:
        r = results[step]
        acc = r['accuracy']
        sec_loss = r['secret_loss']
        l2_loss = r['l2_loss']
        status = "PASS" if acc > 0.5 else "FAIL"
        print(f"{step:6d}   | {acc:8.4f} | {sec_loss:7.4f} | {l2_loss:7.4f} | {status}")

print(f"{'='*80}")

if results:
    all_accs = [r['accuracy'] for r in results.values()]
    final_step = max(results.keys())
    final_acc = results[final_step]['accuracy']
    print(f"Highest accuracy: {max(all_accs):.4f}")
    print(f"Final checkpoint accuracy: {final_acc:.4f}")
    if final_acc > 0.5:
        print("✓ Training successful - model learning detected")
    else:
        print("✗ Training issue - model not learning")
else:
    print("No checkpoints evaluated yet. Training in progress...")
