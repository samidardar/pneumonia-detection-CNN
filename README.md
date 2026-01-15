# 🫁 Pneumonia Detection from Chest X-Ray Images

A deep learning application for detecting pneumonia from chest X-ray images using a trained ResNet18 model with a Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-green.svg)

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 89.26% |
| **Precision** | 86.62% |
| **Recall** | 97.95% |
| **F1-Score** | 91.94% |
| **AUC** | 0.9683 |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web App
```bash
streamlit run pneumonia_app.py
```

### 3. Open in Browser
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
pneumonia_detection/
├── pneumonia_app.py              # Streamlit web application
├── pneumonia_model.py            # Model architecture (TensorFlow/Keras)
├── train_pneumonia.py            # Training script (TensorFlow)
├── train_pneumonia_pytorch.py    # Training script (PyTorch) ✅
├── pneumonia_detection_colab.ipynb  # Google Colab notebook
├── pneumonia_detection_report.md # Detailed project report
├── dataset_info.md               # Dataset information
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── pneumonia_results/            # Training outputs
    ├── pneumonia_model_best.pth  # Best trained model
    ├── pneumonia_model_final.pth # Final trained model
    ├── confusion_matrix.png      # Confusion matrix visualization
    ├── roc_curve.png             # ROC curve
    ├── training_history.png      # Training metrics
    └── evaluation_report.txt     # Evaluation results
```

## 🖥️ Features

- **Drag-and-drop** image upload
- **Real-time** AI predictions
- **Confidence scores** with visual progress bars
- **Model metrics** displayed in sidebar
- **Modern UI** with color-coded results

## 📊 How It Works

1. Upload a chest X-ray image (JPG, JPEG, or PNG)
2. The ResNet18 model processes the image
3. Get instant classification: **NORMAL** or **PNEUMONIA**
4. View confidence percentage and probability distribution

## ⚠️ Disclaimer

This tool is for **educational purposes only**. Always consult a qualified healthcare professional for medical diagnosis.

## 📝 License

MIT License
