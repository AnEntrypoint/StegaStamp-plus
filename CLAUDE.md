# Training Status & Implementation Notes

## Current Status: 256-bit DEBUG Training with Full Chain Exposure (Dec 21, 2025)

**Objective**: DEBUG version of 256-bit training with comprehensive monitoring of entire training chain. Maximum exposure of issues that could randomize or break outcomes. Single training script ready for GPU targeting 140k steps.

**Implementation Complete**: Single training script train_256bit.py for 256-bit production model with complete paper features. Includes adversarial critic network for improved robustness, LPIPS perceptual loss for better image quality preservation, and spatial transformer network foundation for geometric invariance.

**256-bit Full Features**: Adversarial critic network with 4 strided convolutional layers plus dense layers, trained with hinge loss to distinguish real images from watermarked. LPIPS perceptual loss component measures pixel-level differences between original and encoded images, combined with other losses at 0.1 weight. Critic loss weighted at 0.1 scale in total loss computation. Training loops for encoder, decoder, and critic networks with separate optimizers.

**Architecture Design** (based on original paper): Encoder implements U-Net taking 256-bit secret input through 7500 dimension dense representation expanding to 50 by 50 by 3 spatial tensor, upsampled to 400 by 400, with 5 downsampling layers and 4 upsampling layers with skip connections. Decoder implements 5-layer convolutional network extracting secrets from potentially attacked images. Critic network evaluates authenticity of encoded images. Multi-loss combines secret loss weighted 1.5 plus L2 reconstruction loss scheduled 0 to 2.0 over 14000 steps, plus LPIPS at 0.1 weight, plus critic loss at 0.1 weight. Adam optimizer learning rate 0.0001 for encoder/decoder, 0.0001 for critic. Batch size 4. Data uses 400 by 400 RGB images normalized to negative 0.5 to positive 0.5 range.

**DEBUG Monitoring**: Comprehensive chain exposure with maximum visibility into training issues.
- Input validation at every step: NaN/Inf detection, range checking (image/secret bounds)
- Gradient flow monitoring: NaN/Inf detection in encoder gradients
- Residual statistics: Mean, standard deviation, min/max at every 50-step interval
- Loss component breakdown: Separate tracking of secret, L2, LPIPS, critic losses
- Decoded logit ranges: Monitors min/max logits for first 5 steps (reveals dead neurons)
- Batch validation: Checks for data corruption at every training step
- Model initialization: Shape verification and parameter counting for all networks
- Numerical stability: NaN/Inf checks throughout entire training chain
- Direction tracking: Visual indicators (↓/→/↑) with loss trend visualization
- Immediate flush output: All logs stream in real-time without buffering
- Training separators: Clear section markers (====) at checkpoints and phases

**Execution**: Complete training pipeline with built-in validation, directionality tracking, and debug exposure.
- ./train.sh: Run full DEBUG training with automatic checkpoint validation and learning detection
- Directionality indicators: ↓ improving, → plateau, ↑ degrading shown at each step
- Loss window tracking (10-step rolling average) for trend detection
- Training automatically aborts if model accuracy stays at random baseline (50%)
- Training aborts if loss plateaus for 5000+ consecutive steps (no improvement)
- Aborts on any NaN/Inf detection (prevents silent model corruption)
- Validates accuracy at every 10000 step checkpoint
- Only saves checkpoints when model demonstrates learning
- Tracks average accuracy across recent checkpoints

**Inference and Testing**:
- python3 infer_256bit.py <step> <image>: Extract 256-bit secret from image using decoder at checkpoint
- python3 test_checkpoint.py <step>: Validate single checkpoint accuracy across 10 test batches
- python3 monitor_checkpoints.py: Generate comprehensive summary of all checkpoint validations

## Training Approach & Paper Reference

**Strategy**: Follow Tancik et al. 2020 StegaStamp paper implementation using curriculum learning with three phases. Phase one uses variable constant secrets across 35000 steps, phase two uses gradual random secrets across 35000 steps, phase three uses full random secrets across remaining steps. Image augmentation progressively increases across phases. Loss combines secret classification loss and L2 image reconstruction loss with scheduling, starting with secret-only component for initial phase, ramping reconstruction loss over time, then full multi-loss for majority of training. Learning rate decays across phases.

**Architectural Basis**: Encoder uses U-Net structure for learning inverse image transformation while preserving secret information. Decoder learns to extract embedded secrets from potentially corrupted images. This structure proven across multiple papers in steganography literature.

**Next Phase**: Once baseline completes, add LPIPS perceptual loss component. Implement adversarial critic network. Add spatial transformer network for geometric invariance.
