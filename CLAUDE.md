# Training Status & Implementation Notes

## Architecture Limitation Found: Capacity Ceiling at 256-bit (Dec 18, 2025 - 13:40)

**Critical Discovery**: Simple CNN + U-Net both fail at 256-bit despite working at 8-bit

**Tested Architectures**:
1. **Simple 3-layer CNN** (256×256):
   - 8-bit: 68.75% accuracy ✓ (from git history)
   - 64-bit: 55.47% accuracy ✗ (barely above random)
   - 256-bit: 51.56% accuracy ✗ (random guessing)

2. **U-Net with skip connections** (256×256):
   - 256-bit: 51.56% accuracy ✗ (same as simple CNN)
   - Training time: 23.3m (5000 steps)

**Root Cause Analysis**:
- Loss plateaus at **0.693 (log(2)) during random secret phase** regardless of architecture
- This indicates **model learns nothing beyond random guessing** when secrets vary
- Problem is NOT architecture capacity - it's fundamental learning limitation
- Paper (Tancik et al. 2020) uses **140,000 training steps** vs our 5000
- Paper uses **multi-loss scheduling** (message loss + residual loss + critic loss) vs our single BCE loss

**Conclusion**: Scaling from 8-bit (working) to 256-bit (failing) requires:
1. Significantly longer training (140k+ steps, not 5k)
2. Multi-loss with scheduling (not single BCE)
3. Adversarial critic network for robustness
4. Likely requires implementing full paper architecture, not simplified version

**Current Status**: Verified architecture patterns work at small capacity, but don't scale to full 256-bit without additional techniques from paper.

## Previous Results: 8-bit Successfully Trained (Working Baseline)
- **Model**: 256×256, 3-layer encoder + decoder
- **Training**: 2000 steps curriculum (const→gradual→random)
- **Loss**: Constant 0.611 → Gradual 0.609 → Random 0.692
- **Accuracy**: 68.75% on 8-bit secrets ✓
- **Time**: 5.1m
