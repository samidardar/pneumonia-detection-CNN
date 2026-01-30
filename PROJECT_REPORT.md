# 🫁 Pneumonia Detection from Chest X-Ray Images
## Complete Project Report

**Author:** Sami Dardar  
**GitHub:** [github.com/samidardar/pneumonia-detection-CNN](https://github.com/samidardar/pneumonia-detection-CNN)  
**Live Demo:** [pneumonia-detection-cnn.streamlit.app](https://samidardar-pneumonia-detection-cnn-pneumonia-app-xxxxx.streamlit.app)

> [!NOTE]
> Deploy your app at [share.streamlit.io](https://share.streamlit.io) → Login with GitHub → Select `samidardar/pneumonia-detection-CNN` → Main file: `pneumonia_app.py` → Deploy!

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Algorithm & Architecture](#algorithm--architecture)
3. [Dataset](#dataset)
4. [Model Performance](#model-performance)
5. [Training Code Explanation](#training-code-explanation)
6. [Streamlit App Code Explanation](#streamlit-app-code-explanation)
7. [How to Run](#how-to-run)
8. [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

This project implements a **deep learning solution** for detecting pneumonia from chest X-ray images. The system uses a **Convolutional Neural Network (CNN)** based on the **ResNet18 architecture** with **transfer learning** to classify X-ray images as either **NORMAL** or **PNEUMONIA**.

### Key Features
- ✅ **89.26% Accuracy** on test dataset
- ✅ **97.95% Recall** (catches 97.95% of pneumonia cases)
- ✅ **Real-time predictions** via web interface
- ✅ **Modern UI** with confidence visualization

---

## 🧠 Algorithm & Architecture

### Why ResNet18?

**ResNet (Residual Network)** is a powerful CNN architecture that solved the "vanishing gradient problem" using **skip connections**. ResNet18 has 18 layers and is:

- **Pre-trained on ImageNet** (1.2 million images, 1000 classes)
- **Efficient** for medical imaging tasks
- **Fast** inference time for real-time applications

### Transfer Learning Approach

Instead of training from scratch, we use **transfer learning**:

```
┌─────────────────────────────────────────────────────────────┐
│                     ResNet18 Architecture                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Conv1     │ →  │   Layer1    │ →  │   Layer2    │     │
│  │  (Frozen)   │    │  (Frozen)   │    │  (Frozen)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ↓                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Layer3    │ →  │   Layer4    │ →  │  Custom FC  │     │
│  │ (Trainable) │    │ (Trainable) │    │  (New Head) │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Custom Classification Head

We replaced ResNet18's final layer with a custom head for binary classification:

```python
model.fc = nn.Sequential(
    nn.Dropout(0.5),      # Prevent overfitting (50% dropout)
    nn.Linear(512, 256),  # Reduce dimensions: 512 → 256
    nn.ReLU(),            # Activation function
    nn.Dropout(0.3),      # Additional regularization
    nn.Linear(256, 1),    # Output: single probability
    nn.Sigmoid()          # Convert to 0-1 probability
)
```

### Key Concepts Explained

| Concept | What It Does |
|---------|--------------|
| **Dropout** | Randomly "turns off" neurons during training to prevent overfitting |
| **ReLU** | Activation function: `f(x) = max(0, x)` - adds non-linearity |
| **Sigmoid** | Converts output to probability between 0 and 1 |
| **BCELoss** | Binary Cross-Entropy - measures difference between prediction and truth |
| **Adam Optimizer** | Adaptive learning rate optimizer for faster convergence |

---

## 📊 Dataset

**Name:** Chest X-Ray Images (Pneumonia)  
**Source:** [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

### Data Augmentation

To prevent overfitting and improve generalization, we apply these transformations during training:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),        # Resize to model input
    transforms.RandomHorizontalFlip(),    # Flip images horizontally
    transforms.RandomRotation(15),        # Rotate up to ±15 degrees
    transforms.RandomAffine(translate=(0.1, 0.1)),  # Shift by 10%
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary colors
    transforms.ToTensor(),                # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                        [0.229, 0.224, 0.225])    # ImageNet std
])
```

---

## 📈 Model Performance

### Final Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **Accuracy** | 89.26% | Overall correct predictions |
| **Precision** | 86.62% | Of predicted pneumonia, how many are correct |
| **Recall** | 97.95% | Of actual pneumonia cases, how many were detected |
| **F1-Score** | 91.94% | Harmonic mean of precision and recall |
| **AUC** | 0.9683 | Area under ROC curve (1.0 = perfect) |

> [!IMPORTANT]
> **High Recall (97.95%)** is critical in medical diagnosis - we want to catch as many pneumonia cases as possible, even at the cost of some false positives.

### Confusion Matrix

```
                  PREDICTED
              Normal    Pneumonia
         ┌──────────┬──────────┐
 ACTUAL  │          │          │
 Normal  │   175    │    59    │  ← Some normal cases misclassified
         ├──────────┼──────────┤
Pneumonia│     8    │   382    │  ← Very few pneumonia cases missed!
         └──────────┴──────────┘
```

---

## 💻 Training Code Explanation

### File: `train_pneumonia_pytorch.py`

#### 1. Configuration (Lines 45-52)

```python
IMG_SIZE = 224       # Input image size (224×224 pixels)
BATCH_SIZE = 32      # Process 32 images at a time
EPOCHS = 15          # Train for 15 complete passes through data
LEARNING_RATE = 0.001  # How fast the model learns
CLASSES = ['NORMAL', 'PNEUMONIA']  # Our two classes
```

#### 2. Data Loading (Lines 113-119)

```python
# Load images from folders (folder name = class label)
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

# Create data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

#### 3. Model Creation (Lines 141-156)

```python
# Load pre-trained ResNet18
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze early layers (don't update their weights)
for param in list(model.parameters())[:-20]:
    param.requires_grad = False

# Replace final layer for binary classification
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
```

#### 4. Training Loop (Lines 177-240)

```python
for epoch in range(EPOCHS):
    # TRAINING PHASE
    model.train()  # Enable dropout
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass (calculate gradients)
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
    
    # VALIDATION PHASE
    model.eval()  # Disable dropout
    with torch.no_grad():  # Don't calculate gradients
        for images, labels in val_loader:
            outputs = model(images)
            # Calculate validation accuracy
    
    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'pneumonia_model_best.pth')
```

#### 5. Evaluation (Lines 257-286)

```python
# Load best model and evaluate on test set
model.load_state_dict(torch.load('pneumonia_model_best.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = (outputs > 0.5).float()  # Threshold at 0.5
        all_preds.extend(preds)
        all_labels.extend(labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
```

---

## 🌐 Streamlit App Code Explanation

### File: `pneumonia_app.py`

#### 1. Model Loading (Lines 111-134)

```python
@st.cache_resource  # Cache the model (load once, reuse)
def load_model():
    # Recreate the exact same architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    # Load saved weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Set to evaluation mode
    return model
```

#### 2. Image Preprocessing (Lines 137-149)

```python
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Match training size
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],  # ImageNet mean
            [0.229, 0.224, 0.225]   # ImageNet std
        )
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Ensure 3 channels
    
    return transform(image).unsqueeze(0)  # Add batch dimension
```

#### 3. Prediction (Lines 152-162)

```python
def predict(model, device, image):
    with torch.no_grad():  # No gradient calculation needed
        image_tensor = preprocess_image(image).to(device)
        output = model(image_tensor).item()  # Get probability
        
        # output > 0.5 means PNEUMONIA
        prediction = 'PNEUMONIA' if output > 0.5 else 'NORMAL'
        confidence = output if output > 0.5 else 1 - output
        
        return prediction, confidence, output
```

#### 4. Main App Flow (Lines 165-292)

```python
def main():
    # Display header and sidebar with model info
    st.markdown('# 🫁 Pneumonia Detection Scanner')
    
    # File uploader
    uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png'])
    
    if uploaded_file:
        # Show image
        image = Image.open(uploaded_file)
        st.image(image)
        
        # Make prediction
        prediction, confidence, raw = predict(model, device, image)
        
        # Display results with color coding
        if prediction == "NORMAL":
            st.success(f"✅ NORMAL - {confidence*100:.1f}% confident")
        else:
            st.error(f"⚠️ PNEUMONIA - {confidence*100:.1f}% confident")
```

---

## 🚀 How to Run

### Option 1: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/samidardar/pneumonia-detection-CNN.git
cd pneumonia-detection-CNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run pneumonia_app.py
```

### Option 2: Online Demo
**Live URL:** *[Will be added after deployment]*

---

## 🔮 Future Improvements

1. **Multi-class classification** - Detect different types of pneumonia (bacterial vs viral)
2. **Grad-CAM visualization** - Show which parts of the X-ray influenced the decision
3. **Mobile deployment** - Convert to TensorFlow Lite for mobile apps
4. **Larger dataset** - Train on more diverse data for better generalization

---

## 📚 References

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition" - ResNet paper
2. Rajpurkar, P., et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection"
3. Kaggle Chest X-Ray Dataset by Paul Mooney

---

*Report generated for academic presentation purposes. Model is for educational use only.*
