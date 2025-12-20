# StegaStamp-Plus Training Guide

## Current Training Status

**Training is RUNNING** (PID: 29093)
- Process: `python3 train_100bit.py`
- GPU: NVIDIA RTX 3060 (3584 MB allocated)
- Config: 100-bit secrets, 400×400 images, 140,000 steps
- **ETA**: ~35-40 hours total (~140k steps at ~50 steps/min)

## Monitoring Commands

### Quick Status Check
```bash
bash check_training.sh
```
Shows:
- ✓ Process status & resource usage (CPU%, Memory%)
- Latest loss values from training logs
- Checkpoints created so far
- Progress percentage

### View Full Training Logs
```bash
tail -50 /tmp/train_output.log
```
Shows all training output with loss values at each 500-step interval.

### Watch Real-Time Logs (Continuous)
```bash
tail -f /tmp/train_output.log | grep -E "\[|Step|CHECKPOINT"
```
Shows only training milestones as they happen.

### Monitor Background Monitor Script
```bash
tail -f /tmp/monitor.log
```
Real-time milestone tracking with ETA estimates.

## Training Phases

Training has **3 phases** to prevent encoder collapse:

| Phase | Steps | L2 Loss | Purpose |
|-------|-------|---------|---------|
| **Phase 1 (P1)** | 0-14,000 | 0.0 | Secret encoding learning (residuals free to grow) |
| **Phase 2 (P2)** | 14,001-56,000 | 0→2.0 | Gradual imperceptibility (L2 loss ramps up) |
| **Phase 3 (P3)** | 56,001-140,000 | 2.0 | Full loss (both secret encoding & imperceptibility) |

## Understanding the Output

Example training line:
```
[P1] Step  1000/140000 | Secret:0.6234 L2:0.0000 Total:0.9351 | ResidualMag:0.456789 | L2scale:0.0000
```

Breaking it down:
- `[P1]` = Current phase
- `Step 1000/140000` = Progress
- `Secret:0.6234` = Secret loss (target: decrease as training progresses)
- `L2:0.0000` = L2 loss on residual magnitude (Phase 1 = 0)
- `Total:0.9351` = Total loss = Secret×1.5 + L2×L2scale
- `ResidualMag:0.456789` = Encoder output magnitude (should NOT approach 0)
- `L2scale:0.0000` = Current L2 loss weight

## Critical Checkpoints to Monitor

These are the key milestones to verify training is working:

| Step | What to Check | Success Criteria |
|------|---------------|------------------|
| **500** | First loss plateau check | Secret loss < 0.69, ResidualMag > 0.01 |
| **1,000** | Phase 1 continues | Secret loss decreasing, L2 scale still 0.0 |
| **5,000** | Mid-phase 1 | Consistent learning pattern |
| **10,000** | First checkpoint | Model saved, >55% inference accuracy expected |
| **14,000** | Phase 1→2 transition | L2 loss starts ramping |
| **56,000** | Phase 2→3 transition | L2 loss reaches maximum (2.0) |
| **140,000** | Complete | Final model trained |

## Testing a Checkpoint

Once a checkpoint is created (e.g., step 10,000):

```bash
export LD_LIBRARY_PATH="/home/user/diffusers/pixel_art_venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.6"
python3 test_100bit_inference.py 10000
```

Output shows average accuracy across 20 test cases:
- **50%** = Random baseline (not learning)
- **55-60%** = Early learning (good at 10k steps)
- **70%+** = Strong learning (expected at 50k+ steps)
- **>80%** = Excellent (expected at final training)

## Files in Use

| File | Purpose |
|------|---------|
| `train_100bit.py` | Main training script (RUNNING) |
| `train_100bit_gpu.sh` | GPU environment setup |
| `check_training.sh` | Status check command |
| `test_100bit_inference.py` | Inference testing |
| `monitor_training.py` | Background monitoring daemon |
| `/tmp/train_output.log` | Training output log |
| `/tmp/monitor.log` | Monitor output log |
| `encoder_100bit_step_*.keras` | Encoder checkpoints |
| `decoder_100bit_step_*.keras` | Decoder checkpoints |

## Troubleshooting

### Training seems stuck
Check logs: `tail -20 /tmp/train_output.log`
- If Secret loss stays at 0.6931 → Encoder not learning
- If ResidualMag → 0 → Phase 1 L2=0 not working

### Checkpoint creation failed
Check disk space: `df -h /home/user/StegaStamp-plus/`
Each checkpoint pair: ~540MB total

### GPU memory issues
Current memory usage: 3.5GB / 6.1GB
If OOM: Reduce BATCH_SIZE in train_100bit.py (currently 4)

## Next Steps After Training

1. **When first checkpoint (10k) appears**: Test accuracy
2. **After Phase 1 complete (14k steps)**: L2 scaling starts
3. **At 50% (70k steps)**: Halfway through
4. **When complete (140k steps)**:
   - Run inference test on final model
   - Add LPIPS perceptual loss for next iteration
   - Implement GAN critic for robustness
   - Add STN for geometric invariance

## Key Success Indicators

✓ **Phase 1 (0-14k)**: Secret loss decreasing, L2=0, ResidualMag stable
✓ **Phase 2 (14k-56k)**: L2 loss gradually increasing, Secret loss continues improving
✓ **Phase 3 (56k-140k)**: Both losses converge, model reaches final capacity
✓ **Final checkpoint**: Inference accuracy >70% on random test cases

