# Training Status & Implementation Notes

## Current Status: 100-bit and 256-bit Training Ready (Dec 21, 2025)

**Objective**: Both 100-bit and 256-bit training implementations complete and ready for execution on GPU system. Following Tancik et al. 2020 StegaStamp paper targeting 140k steps for full training.

**Implementation Complete**: Codebase cleaned to contain only production files. Duplicate test/debug/documentation files removed. Two training scripts ready: train_100bit.py for baseline (100-bit secrets) and train_256bit.py for full model (256-bit secrets). Both implement identical U-Net encoder architecture with multi-loss training combining secret classification loss plus L2 image reconstruction loss with scheduling.

**Architecture Design** (based on original paper): Encoder implements U-Net taking secret input through 7500 dimension dense representation expanding to 50 by 50 by 3 spatial tensor, upsampled to 400 by 400, with 5 downsampling layers and 4 upsampling layers with skip connections. Decoder implements 5-layer convolutional network with 2 by 2 strides, flattening layer, 512 dimension dense layer, then outputs secret logits (100 for train_100bit.py, 256 for train_256bit.py). Multi-loss combines secret loss weighted 1.5 plus L2 loss with scheduling ramping from 0 to 2.0 over first 14000 steps. Optimizer uses Adam with learning rate 0.0001. Batch size 4 images per step. Data uses 400 by 400 RGB images normalized to negative 0.5 to positive 0.5 range.

**Execution**: Run using train.sh wrapper script. Usage: ./train.sh 100 for 100-bit baseline or ./train.sh 256 for 256-bit full model. Both scripts save checkpoints every 10000 steps and final model after 140000 steps.

## Training Approach & Paper Reference

**Strategy**: Follow Tancik et al. 2020 StegaStamp paper implementation using curriculum learning with three phases. Phase one uses variable constant secrets across 35000 steps, phase two uses gradual random secrets across 35000 steps, phase three uses full random secrets across remaining steps. Image augmentation progressively increases across phases. Loss combines secret classification loss and L2 image reconstruction loss with scheduling, starting with secret-only component for initial phase, ramping reconstruction loss over time, then full multi-loss for majority of training. Learning rate decays across phases.

**Architectural Basis**: Encoder uses U-Net structure for learning inverse image transformation while preserving secret information. Decoder learns to extract embedded secrets from potentially corrupted images. This structure proven across multiple papers in steganography literature.

**Next Phase**: Once baseline completes, add LPIPS perceptual loss component. Implement adversarial critic network. Add spatial transformer network for geometric invariance.
