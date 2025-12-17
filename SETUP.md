# StegaStamp WebGPU Browser Setup

This guide explains how to set up and run the browser-based StegaStamp encoder/decoder.

## Prerequisites

- Node.js 18+
- Python 3.8+ (for model conversion)
- TensorFlow 2.x or compatible Python environment

## Installation

1. Install Node dependencies:
```bash
npm install
```

2. Download the pretrained models:
```bash
bash scripts/download-models.sh
```

3. Convert TensorFlow models to ONNX:
```bash
python3 scripts/convert-models.py
```

## Development

Run the dev server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## Building for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/
│   ├── Encoder.tsx    - Encoding interface
│   ├── Decoder.tsx    - Decoding interface
│   └── Detector.tsx   - Detection interface (optional)
├── models/
│   └── StegaStampModel.ts - ONNX model loader
├── utils/
│   ├── bch.ts         - BCH error correction
│   └── imageProcessing.ts - Image utilities
├── hooks/
│   └── useStegaStamp.ts - React hook for models
├── App.tsx            - Main app component
└── main.tsx           - Entry point

public/models/        - ONNX model files (generated)
```

## Usage

### Encoding
1. Go to "Encoder" tab
2. Upload an image (JPG/PNG)
3. Enter a secret message (max 7 chars)
4. Click "Encode"
5. Download the encoded image

### Decoding
1. Go to "Decoder" tab
2. Upload an encoded image
3. Click "Decode"
4. View the recovered message

## Architecture

- **Encoder Network**: Encodes secret bits into image while maintaining visual similarity
- **Decoder Network**: Extracts bits from potentially corrupted images
- **Detector Network**: Locates and rectifies StegaStamps at various angles
- **BCH Error Correction**: Recovers bits even with corruption
- **ONNX Runtime**: GPU-accelerated inference in browser

## Notes

- Models must be converted to ONNX format for browser execution
- ONNX Runtime uses WASM for CPU execution (WebGPU support requires additional setup)
- Image dimensions are fixed to 224x224 for model inputs
- Secret messages are padded to 100 bits, with 44 bits of error correction (56 bits effective)
