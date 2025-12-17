# Model Setup Guide

## The StegaStamp-plus Approach

**StegaStamp-plus** (this repository) is designed primarily for **training your own models** in Google Colab. Unlike the original repository which had downloadable pretrained models, this fork emphasizes:

1. **Colab-first workflow** - Full training notebook included
2. **Complete source code** - All training scripts and hyperparameters
3. **Reproducible training** - Improved from original repo with documentation
4. **Your own models** - Train with custom datasets

## Getting Started: Train Models in Colab

### Recommended: Google Colab (Free GPU)

The easiest path is training in Google Colab with free GPU access:

1. **Open the Colab notebook**:
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Charmve/StegaStamp-plus/blob/master/StegaStamp_train_model.ipynb)

2. **The notebook provides**:
   - TensorFlow 1.13 installation
   - Dataset setup instructions
   - Full training pipeline
   - Model saving to Google Drive
   - Complete encode/decode examples

3. **Training process**:
   - Prepare/download your dataset
   - Mount Google Drive
   - Run training (typically 4-8 hours on Colab GPU)
   - Models save to `saved_models/stegastamp_pretrained`
   - Download models to your local machine

### Alternative: Train Locally

If you prefer local training:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set dataset path in train.py
# TRAIN_PATH = /path/to/your/dataset

# Start training
bash scripts/base.sh my_experiment_name

# Models saved to: saved_models/my_experiment_name/
```

**Requirements**:
- NVIDIA GPU (CUDA 10.0+)
- 32GB+ RAM recommended
- TensorFlow 1.13.1
- Training time: 4-8+ hours

## Expected Model Structure

Once obtained, models should be organized as:
```
models/
├── saved_models/
│   └── stegastamp_pretrained/
│       ├── saved_model.pb
│       ├── variables/
│       │   ├── variables.data-00000-of-00001
│       │   └── variables.index
│       └── assets/
└── detector_models/
    └── stegastamp_detector/
        ├── saved_model.pb
        ├── variables/
        │   ├── variables.data-00000-of-00001
        │   └── variables.index
        └── assets/
```

## Converting Models to Browser Format

Once you have trained models (from Colab or local training), convert them to ONNX for browser use:

### Step 1: Place Models

```bash
# Copy trained models to models/ directory
models/
├── saved_models/
│   └── stegastamp_pretrained/
│       ├── saved_model.pb
│       └── variables/
└── detector_models/
    └── stegastamp_detector/
        ├── saved_model.pb
        └── variables/
```

### Step 2: Install Conversion Dependencies

```bash
pip install tensorflow==1.13.1 tf2onnx onnx
```

### Step 3: Convert to ONNX

```bash
python3 scripts/convert-models.py
```

This creates:
- `public/models/encoder.onnx`
- `public/models/detector.onnx`

### Step 4: Run Web Interface

```bash
npm run dev
# Browser models automatically load from public/models/
```

## Complete Workflow

```bash
# 1. Train in Colab (see above)
# 2. Download models from Google Drive
# 3. Extract to models/ directory
# 4. Install Python dependencies
pip install -r requirements.txt
pip install tf2onnx

# 5. Convert models
python3 scripts/convert-models.py

# 6. Install web dependencies
npm install

# 7. Run dev server
npm run dev

# 8. Open http://localhost:5173
```

## Model File Sizes

- **Encoder SavedModel**: ~50MB
- **Detector SavedModel**: ~30MB
- **Encoder ONNX**: ~40MB
- **Detector ONNX**: ~25MB
- **Browser build (with ONNX RT)**: ~24MB total (gzipped: ~5.7MB)

## Troubleshooting

**Models not loading in browser?**
- Check browser console for ONNX Runtime errors
- Verify `public/models/` contains `.onnx` files
- Try clearing browser cache

**Conversion failed?**
- Ensure TensorFlow 1.13 compatibility
- Check model SavedModel.pb files exist
- Try manual conversion: `python -m tf2onnx.convert --saved-model models/saved_models/stegastamp_pretrained --output public/models/encoder.onnx --opset 13`

See SETUP.md for general installation instructions.
