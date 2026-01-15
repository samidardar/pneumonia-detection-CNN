# Dataset Information: Chest X-Ray Images (Pneumonia)

## Overview

This document provides instructions for obtaining and preparing the Kaggle Chest X-Ray Pneumonia dataset for training the pneumonia detection model.

## Dataset Details

| Property | Value |
|----------|-------|
| **Name** | Chest X-Ray Images (Pneumonia) |
| **Source** | Kaggle / Mendeley Data |
| **Classes** | 2 (Normal, Pneumonia) |
| **Total Images** | 5,856 |
| **Total Size** | ~1.15 GB |

### Split Distribution

| Split | Images | Size |
|-------|--------|------|
| Training | 5,216 | ~1.07 GB |
| Validation | 16 | ~42.8 MB |
| Testing | 624 | ~35.4 MB |

## Download Instructions

### Option 1: Kaggle CLI (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials
# Download kaggle.json from https://www.kaggle.com/account
# Place in ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<user>\.kaggle\kaggle.json (Windows)

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract
unzip chest-xray-pneumonia.zip -d chest-xray-pneumonia
```

### Option 2: Direct Download

1. Visit: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
2. Click "Download" (requires Kaggle account)
3. Extract the ZIP file

### Option 3: Mendeley Data (Original Source)

Download from: https://data.mendeley.com/datasets/rscbjbr9sj/2

## Dataset Structure

```
chest_xray/
├── train/
│   ├── NORMAL/
│   │   └── *.jpeg (1,341 images)
│   └── PNEUMONIA/
│       └── *.jpeg (3,875 images)
├── val/
│   ├── NORMAL/
│   │   └── *.jpeg (8 images)
│   └── PNEUMONIA/
│       └── *.jpeg (8 images)
└── test/
    ├── NORMAL/
    │   └── *.jpeg (234 images)
    └── PNEUMONIA/
        └── *.jpeg (390 images)
```

## Usage with Model

```bash
# Train the model
python pneumonia_model.py --data_dir path/to/chest_xray

# Train with fine-tuning
python pneumonia_model.py --data_dir path/to/chest_xray --fine_tune

# Predict single image
python pneumonia_model.py --predict path/to/xray.jpg --model_path pneumonia_model.h5
```

## Data Augmentation

The training pipeline applies the following augmentations:
- **Rotation**: ±20 degrees
- **Width/Height Shift**: ±20%
- **Shear**: 20%
- **Zoom**: ±20%
- **Horizontal Flip**: Yes
- **Brightness**: 80-120%

## Citation

If using this dataset, please cite:

```
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence 
Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, V2, 
doi: 10.17632/rscbjbr9sj.2
```

**Original Paper**: [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) (Cell, 2018)
