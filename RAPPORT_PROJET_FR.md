# 🫁 Détection de Pneumonie à partir d'Images Radiographiques
## Rapport de Projet Complet

**Auteur:** Sami Dardar  
**GitHub:** [github.com/samidardar/pneumonia-detection-CNN](https://github.com/samidardar/pneumonia-detection-CNN)  
**Démo en ligne:** [pneumonia-detection-cnn.streamlit.app](https://share.streamlit.io)

> [!NOTE]
> Déployez l'application sur [share.streamlit.io](https://share.streamlit.io) → Connexion GitHub → Sélectionnez `samidardar/pneumonia-detection-CNN` → Fichier: `pneumonia_app.py` → Déployer!

---

## 📋 Table des Matières

1. [Aperçu du Projet](#aperçu-du-projet)
2. [Algorithme et Architecture](#algorithme-et-architecture)
3. [Jeu de Données](#jeu-de-données)
4. [Performance du Modèle](#performance-du-modèle)
5. [Explication du Code d'Entraînement](#explication-du-code-dentraînement)
6. [Explication de l'Application Streamlit](#explication-de-lapplication-streamlit)
7. [Comment Exécuter](#comment-exécuter)

---

## 🎯 Aperçu du Projet

Ce projet implémente une solution d'**apprentissage profond** pour détecter la pneumonie à partir de radiographies thoraciques. Le système utilise un **Réseau de Neurones Convolutif (CNN)** basé sur l'architecture **ResNet18** avec **apprentissage par transfert** pour classifier les images en **NORMAL** ou **PNEUMONIE**.

### Caractéristiques Principales
- ✅ **89.26% de Précision** sur le jeu de test
- ✅ **97.95% de Rappel** (détecte 97.95% des cas de pneumonie)
- ✅ **Prédictions en temps réel** via interface web
- ✅ **Interface moderne** avec visualisation de la confiance

---

## 🧠 Algorithme et Architecture

### Pourquoi ResNet18?

**ResNet (Réseau Résiduel)** est une architecture CNN puissante qui a résolu le "problème du gradient qui disparaît" grâce aux **connexions de saut**. ResNet18 possède 18 couches et est:

- **Pré-entraîné sur ImageNet** (1.2 million d'images, 1000 classes)
- **Efficace** pour les tâches d'imagerie médicale
- **Rapide** pour les prédictions en temps réel

### Approche d'Apprentissage par Transfert

Au lieu d'entraîner depuis zéro, nous utilisons l'**apprentissage par transfert**:

```
┌─────────────────────────────────────────────────────────────┐
│                   Architecture ResNet18                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Conv1     │ →  │   Couche1   │ →  │   Couche2   │     │
│  │   (Gelée)   │    │   (Gelée)   │    │   (Gelée)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ↓                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Couche3   │ →  │   Couche4   │ →  │  FC Perso.  │     │
│  │ (Entraînable)│   │ (Entraînable)│   │(Nouvelle Tête)│    │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Tête de Classification Personnalisée

Nous avons remplacé la couche finale de ResNet18 par une tête personnalisée:

```python
model.fc = nn.Sequential(
    nn.Dropout(0.5),      # Évite le surapprentissage (50% dropout)
    nn.Linear(512, 256),  # Réduit dimensions: 512 → 256
    nn.ReLU(),            # Fonction d'activation
    nn.Dropout(0.3),      # Régularisation supplémentaire
    nn.Linear(256, 1),    # Sortie: probabilité unique
    nn.Sigmoid()          # Convertit en probabilité 0-1
)
```

### Concepts Clés Expliqués

| Concept | Fonction |
|---------|----------|
| **Dropout** | Désactive aléatoirement des neurones pendant l'entraînement pour éviter le surapprentissage |
| **ReLU** | Fonction d'activation: `f(x) = max(0, x)` - ajoute la non-linéarité |
| **Sigmoid** | Convertit la sortie en probabilité entre 0 et 1 |
| **BCELoss** | Entropie Croisée Binaire - mesure la différence entre prédiction et vérité |
| **Adam** | Optimiseur à taux d'apprentissage adaptatif |

---

## 📊 Jeu de Données

**Nom:** Chest X-Ray Images (Pneumonia)  
**Source:** [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Division | Normal | Pneumonie | Total |
|----------|--------|-----------|-------|
| Entraînement | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

### Augmentation de Données

Pour éviter le surapprentissage, nous appliquons ces transformations:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),        # Redimensionner
    transforms.RandomHorizontalFlip(),    # Retournement horizontal
    transforms.RandomRotation(15),        # Rotation ±15 degrés
    transforms.RandomAffine(translate=(0.1, 0.1)),  # Décalage 10%
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variation couleurs
    transforms.ToTensor(),                # Conversion en tenseur
    transforms.Normalize([0.485, 0.456, 0.406],   # Moyenne ImageNet
                        [0.229, 0.224, 0.225])    # Écart-type ImageNet
])
```

---

## 📈 Performance du Modèle

### Métriques Finales

| Métrique | Valeur | Signification |
|----------|--------|---------------|
| **Précision (Accuracy)** | 89.26% | Prédictions correctes globales |
| **Précision (Precision)** | 86.62% | Parmi les prédictions "pneumonie", combien sont correctes |
| **Rappel (Recall)** | 97.95% | Parmi les vrais cas de pneumonie, combien détectés |
| **Score F1** | 91.94% | Moyenne harmonique précision/rappel |
| **AUC** | 0.9683 | Aire sous courbe ROC (1.0 = parfait) |

> [!IMPORTANT]
> **Rappel élevé (97.95%)** est crucial en diagnostic médical - nous voulons détecter autant de cas de pneumonie que possible.

### Matrice de Confusion

```
                  PRÉDIT
              Normal    Pneumonie
         ┌──────────┬──────────┐
 RÉEL    │          │          │
 Normal  │   175    │    59    │  ← Quelques cas normaux mal classés
         ├──────────┼──────────┤
Pneumonie│     8    │   382    │  ← Très peu de pneumonies ratées!
         └──────────┴──────────┘
```

---

## 💻 Explication du Code d'Entraînement

### Fichier: `train_pneumonia_pytorch.py`

#### 1. Configuration

```python
IMG_SIZE = 224       # Taille image (224×224 pixels)
BATCH_SIZE = 32      # Traiter 32 images à la fois
EPOCHS = 15          # 15 passages complets sur les données
LEARNING_RATE = 0.001  # Vitesse d'apprentissage
CLASSES = ['NORMAL', 'PNEUMONIA']
```

#### 2. Création du Modèle

```python
# Charger ResNet18 pré-entraîné
model = models.resnet18(weights='IMAGENET1K_V1')

# Geler les premières couches
for param in list(model.parameters())[:-20]:
    param.requires_grad = False

# Remplacer la dernière couche
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
```

#### 3. Boucle d'Entraînement

```python
for epoch in range(EPOCHS):
    # PHASE D'ENTRAÎNEMENT
    model.train()  # Active le dropout
    for images, labels in train_loader:
        # Passage avant
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Passage arrière (calcul des gradients)
        loss.backward()
        
        # Mise à jour des poids
        optimizer.step()
        optimizer.zero_grad()
    
    # Sauvegarder le meilleur modèle
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'pneumonia_model_best.pth')
```

---

## 🌐 Explication de l'Application Streamlit

### Fichier: `pneumonia_app.py`

#### 1. Chargement du Modèle

```python
@st.cache_resource  # Mettre en cache (charger une fois)
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(...)  # Même architecture
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Mode évaluation
    return model
```

#### 2. Prétraitement d'Image

```python
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)
```

#### 3. Prédiction

```python
def predict(model, device, image):
    with torch.no_grad():
        output = model(preprocess_image(image)).item()
        
        # output > 0.5 signifie PNEUMONIE
        prediction = 'PNEUMONIE' if output > 0.5 else 'NORMAL'
        confidence = output if output > 0.5 else 1 - output
        
        return prediction, confidence
```

---

## 🚀 Comment Exécuter

### Option 1: Exécution Locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/samidardar/pneumonia-detection-CNN.git
cd pneumonia-detection-CNN

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run pneumonia_app.py
```

### Option 2: Démo en Ligne
Visitez [share.streamlit.io](https://share.streamlit.io) et déployez depuis GitHub!

---

## 📚 Références

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
2. Rajpurkar, P., et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection"
3. Kaggle Chest X-Ray Dataset par Paul Mooney

---

*Rapport généré à des fins de présentation académique. Le modèle est uniquement à but éducatif.*
