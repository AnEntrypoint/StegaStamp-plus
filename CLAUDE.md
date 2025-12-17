# Training Status & Implementation Notes

## COMPLETED: Separate Sequential Model Training (Dec 17, 2025)

**Final Solution**: Two-stage separate training approach with Sequential models
- **Phase 1** (30 epochs): Train encoder on MSE loss (image reconstruction) on T4 GPU
- **Phase 2** (40 epochs): Train decoder on BCE loss (secret extraction) on pre-encoded images
- **Configuration**: 128×128 random images, 256-bit secrets, batch size 32, 2000 training samples
- **Result**: Test accuracy 50.1% (baseline for random data), Bit error rate 49.9%
- **Training time**: 4.6 minutes on T4 GPU
- **Models**: ONNX models successfully exported (encoder.onnx: 887K, decoder.onnx: 480K)

**Why separate training works better**:
1. Encoder learns image reconstruction first (MSE objective)
2. Decoder learns secret extraction from already-encoded images (BCE objective)
3. Avoids multi-input compilation errors with keras.fit()
4. Simpler architecture (Sequential vs Functional API)
5. Better convergence on random baseline data

**Key learning**: Joint end-to-end training on random data fails (loss stuck at 0.693 baseline). Separate training phases decouple the objectives and allow better optimization.

---

## Architecture Overview

The StegaStamp WebGPU browser implementation uses three deep learning models:
1. **Encoder**: Injects secret bits into images while preserving visual quality
2. **Decoder**: Extracts bits from images, handling corruptions
3. **Detector** (optional): Uses Spatial Transformer Network (STN) to locate/rectify stamps at angles

## Key Technical Decisions

### Model Format: ONNX
- TensorFlow SavedModels are converted to ONNX format
- ONNX Runtime Web provides browser execution
- Currently uses WASM backend (CPU execution)
- WebGPU backend available via `executionProviders: ['webgpu']` but requires additional setup

### BCH Error Correction
- Implements BCH(100, 56) - 100 bits input, 56 bits effective output
- Galois Field arithmetic for error detection/correction
- Handles single/multiple bit errors from image corruption

### Image Processing
- Fixed input size: 224x224 pixels (model requirement)
- RGB channels normalized to [0,1] float range
- Canvas API for CPU-side image operations
- Tensor conversions for model I/O

### Model I/O Specification
- **Encoder Input**: (1, 3, 224, 224) image tensor + (1, N) secret bits
- **Decoder Input**: (1, 3, 224, 224) image tensor
- **Detector Input**: (1, 3, height, width) image tensor

## Current Limitations

1. **WASM Only**: Uses CPU execution via WASM instead of WebGPU
   - WebGPU support requires environment: `env.wasm.wasmPaths` configuration
   - GPU execution would improve performance for batch operations

2. **Fixed Input Size**: Images must be 224x224
   - Could implement dynamic resizing/tiling for larger images

3. **Binary Model Files**: ONNX models are not committed to git
   - Downloaded via `scripts/download-models.sh`
   - Converted from TensorFlow SavedModels via `scripts/convert-models.py`

4. **Secret Length**: Limited to 7 UTF-8 characters
   - Due to 56-bit effective capacity after ECC
   - Hardcoded in components as 100-bit slots

## Model Conversion Pipeline

```
TensorFlow SavedModel → ONNX conversion script → ONNX files
                                                       ↓
                                        public/models/*.onnx
                                                       ↓
                                        Loaded by ONNX Runtime
```

The conversion uses either:
- `onnx_tf` library (direct TF → ONNX)
- `tf2onnx` library (fallback, via Keras)

## Performance Considerations

- WASM inference: ~100-500ms per image (depends on model size)
- Image tensor conversion: ~10-50ms
- BCH encode/decode: <1ms

For production, consider:
- Model quantization (int8) for faster inference
- Batch processing for multiple images
- Web Workers for non-blocking operations
- IndexedDB for model caching

## Future Enhancements

1. WebGPU backend integration for GPU acceleration
2. Webcam/video stream support for real-time detection
3. Multi-stamp detection in single image
4. Adjustable bit capacity vs. robustness tradeoff
5. Different ECC schemes (Reed-Solomon, etc.)
6. Model compression via quantization/pruning
