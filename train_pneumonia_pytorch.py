# Pneumonia Detection Model Training Script (PyTorch)
# =====================================================
# Using PyTorch + torchvision for compatibility with Windows

import os
import sys
import numpy as np
from datetime import datetime

print("=" * 60)
print("   PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES")
print("   Training with ResNet18 Transfer Learning (PyTorch)")
print("=" * 60)
print()

# Install required packages
def install_packages():
    packages = ['torch', 'torchvision', 'pillow', 'matplotlib', 'seaborn', 'scikit-learn', 'tqdm']
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            print(f"Installing {pkg}...")
            os.system(f'pip install {pkg} --quiet')

install_packages()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

print(f"PyTorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# CONFIGURATION
# =============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
CLASSES = ['NORMAL', 'PNEUMONIA']

# Dataset path
DATASET_PATH = r"C:\Users\PC\Downloads\archive (6)"

# Find data directory
DATA_DIR = None
for root, dirs, files in os.walk(DATASET_PATH):
    if 'train' in dirs and 'test' in dirs:
        DATA_DIR = root
        break
    for d in dirs:
        if d == 'chest_xray':
            potential = os.path.join(root, d)
            if os.path.exists(os.path.join(potential, 'train')):
                DATA_DIR = potential
                break

if DATA_DIR is None:
    if os.path.exists(os.path.join(DATASET_PATH, 'train')):
        DATA_DIR = DATASET_PATH
    else:
        print(f"ERROR: Could not find dataset. Searched in: {DATASET_PATH}")
        sys.exit(1)

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

OUTPUT_DIR = r"c:\Users\PC\Downloads\New folder\pneumonia_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📁 Dataset: {DATA_DIR}")
print(f"📂 Output: {OUTPUT_DIR}")

# =============================================================================
# DATA TRANSFORMS
# =============================================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =============================================================================
# DATA LOADING
# =============================================================================
print("\n" + "=" * 50)
print("   LOADING DATA")
print("=" * 50)

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=test_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")
print(f"✅ Test samples: {len(test_dataset)}")
print(f"✅ Classes: {train_dataset.classes}")

# Class weights for imbalanced data
class_counts = [0, 0]
for _, label in train_dataset:
    class_counts[label] += 1
total = sum(class_counts)
class_weights = torch.tensor([total / (2 * c) for c in class_counts]).to(device)
print(f"📊 Class weights: {class_weights.tolist()}")

# =============================================================================
# MODEL
# =============================================================================
print("\n" + "=" * 50)
print("   BUILDING MODEL")
print("=" * 50)

model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze early layers
for param in list(model.parameters())[:-20]:
    param.requires_grad = False

# Replace final FC layer
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

model = model.to(device)
print(f"✅ Model: ResNet18 (pretrained)")
print(f"✅ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

# =============================================================================
# TRAINING
# =============================================================================
print("\n" + "=" * 50)
print("   TRAINING")
print("=" * 50)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        predicted = (outputs > 0.5).float()
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
    
    train_loss = train_loss / train_total
    train_acc = train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'pneumonia_model_best.pth'))
        print(f"   ✅ Saved best model (val_acc: {val_acc:.4f})")

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "=" * 50)
print("   EVALUATION")
print("=" * 50)

# Load best model
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'pneumonia_model_best.pth')))
model.eval()

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        outputs = model(images).squeeze()
        
        probs = outputs.cpu().numpy()
        preds = (outputs > 0.5).float().cpu().numpy()
        
        all_probs.extend(probs if len(probs.shape) > 0 else [probs.item()])
        all_preds.extend(preds if len(preds.shape) > 0 else [preds.item()])
        all_labels.extend(labels.numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_prob = np.array(all_probs)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n" + "=" * 50)
print("         FINAL RESULTS")
print("=" * 50)
print(f"  ✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ✅ Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  ✅ Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  ✅ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print("=" * 50)

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 50)
print("   SAVING RESULTS")
print("=" * 50)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={'size': 20})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print("✅ Saved: confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=150)
plt.close()
print("✅ Saved: roc_curve.png")

# Training History
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Validation')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'], label='Validation')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)
plt.close()
print("✅ Saved: training_history.png")

# Save model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'pneumonia_model_final.pth'))
print("✅ Saved: pneumonia_model_final.pth")

# Evaluation report
with open(os.path.join(OUTPUT_DIR, 'evaluation_report.txt'), 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("    PNEUMONIA DETECTION MODEL - EVALUATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: ResNet18 (PyTorch)\n\n")
    f.write("PERFORMANCE METRICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"AUC:       {roc_auc:.4f}\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 40 + "\n")
    f.write(classification_report(y_true, y_pred, target_names=CLASSES))
print("✅ Saved: evaluation_report.txt")

print("\n" + "=" * 60)
print("   🎉 TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📁 All results saved to: {OUTPUT_DIR}")
print("\nFiles created:")
print("   - pneumonia_model_final.pth (trained model)")
print("   - pneumonia_model_best.pth (best validation model)")
print("   - confusion_matrix.png")
print("   - roc_curve.png")
print("   - training_history.png")
print("   - evaluation_report.txt")
