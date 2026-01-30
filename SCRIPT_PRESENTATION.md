# 🫁 DÉTECTION DE PNEUMONIE PAR INTELLIGENCE ARTIFICIELLE
## Script de Présentation Complet

**Projet:** Détection Automatique de Pneumonie à partir de Radiographies Thoraciques  
**Auteur:** Sami Dardar  
**Date:** Janvier 2026

---

# 📖 PARTIE 1: INTRODUCTION (2-3 minutes)

## Slide 1: Titre et Contexte

**À LIRE:**

> "Bonjour à tous. Aujourd'hui, je vais vous présenter mon projet de détection de pneumonie par intelligence artificielle.
>
> La pneumonie est une infection pulmonaire grave qui touche des millions de personnes chaque année. C'est l'une des principales causes de mortalité infantile dans le monde. Le diagnostic précoce est crucial, mais il nécessite l'interprétation de radiographies thoraciques par des médecins spécialisés, ce qui n'est pas toujours disponible dans les régions défavorisées.
>
> Mon projet propose une solution: utiliser l'intelligence artificielle pour analyser automatiquement les radiographies et détecter la présence de pneumonie avec une précision de près de 90%."

---

## Slide 2: Objectifs du Projet

**À LIRE:**

> "Les objectifs de ce projet sont les suivants:
>
> **Premièrement**, développer un modèle d'apprentissage profond capable de classifier les radiographies thoraciques en deux catégories: NORMAL ou PNEUMONIE.
>
> **Deuxièmement**, atteindre une performance élevée, en particulier un rappel élevé, car il est crucial de ne pas manquer de vrais cas de pneumonie.
>
> **Troisièmement**, créer une interface web accessible permettant à n'importe qui de tester le modèle en téléchargeant une image.
>
> **Quatrièmement**, déployer l'application en ligne pour qu'elle soit accessible partout dans le monde."

---

# 📖 PARTIE 2: TECHNOLOGIE UTILISÉE (5-7 minutes)

## Slide 3: Qu'est-ce que l'Apprentissage Profond?

**À LIRE:**

> "Avant d'entrer dans les détails techniques, permettez-moi d'expliquer ce qu'est l'apprentissage profond.
>
> L'apprentissage profond, ou Deep Learning en anglais, est une branche de l'intelligence artificielle qui s'inspire du fonctionnement du cerveau humain. Il utilise des réseaux de neurones artificiels composés de plusieurs couches - d'où le terme 'profond'.
>
> Chaque couche apprend à reconnaître des caractéristiques de plus en plus complexes. Par exemple:
> - La première couche peut détecter des bords et des contours
> - La deuxième couche combine ces bords pour former des formes
> - Les couches suivantes reconnaissent des textures, des motifs
> - Les dernières couches identifient des objets complets
>
> Dans notre cas, le réseau apprend à reconnaître les signes visuels de la pneumonie dans les radiographies."

---

## Slide 4: Les Réseaux de Neurones Convolutifs (CNN)

**À LIRE:**

> "Pour analyser des images, on utilise un type spécial de réseau appelé CNN - Réseau de Neurones Convolutif.
>
> Le mot 'convolutif' vient de l'opération mathématique de convolution. Imaginez un petit filtre qui se déplace sur l'image et qui détecte des motifs spécifiques à chaque position.
>
> Un CNN est composé de plusieurs types de couches:
>
> **Les couches de convolution**: Elles appliquent des filtres pour détecter des caractéristiques comme les bords, les textures, ou les formes.
>
> **Les couches de pooling**: Elles réduisent la taille de l'image tout en gardant l'information importante. C'est comme faire un zoom arrière.
>
> **Les couches entièrement connectées**: À la fin du réseau, elles prennent toutes les caractéristiques détectées et font la classification finale."

---

## Slide 5: Pourquoi ResNet18?

**À LIRE:**

> "Pour ce projet, j'ai choisi l'architecture ResNet18. Laissez-moi vous expliquer pourquoi.
>
> ResNet signifie 'Réseau Résiduel'. Il a été créé par Microsoft Research en 2015 et a révolutionné l'apprentissage profond en résolvant un problème majeur: le problème du gradient qui disparaît.
>
> Dans les réseaux très profonds, quand on entraîne le modèle, le signal d'erreur doit se propager à travers toutes les couches. Dans les anciens réseaux, ce signal s'affaiblissait tellement qu'il ne permettait plus d'entraîner les premières couches.
>
> ResNet résout ce problème avec des 'connexions de saut'. Ces connexions permettent au signal de passer directement d'une couche à une autre plus loin, sans s'affaiblir.
>
> Le '18' dans ResNet18 indique que le réseau a 18 couches. C'est un bon compromis entre performance et vitesse - assez puissant pour notre tâche, mais pas trop lourd à exécuter."

---

## Slide 6: L'Apprentissage par Transfert

**À LIRE:**

> "Un autre concept clé de ce projet est l'apprentissage par transfert.
>
> Entraîner un réseau de neurones depuis zéro nécessite des millions d'images et des semaines de calcul. Nous n'avons pas autant de radiographies médicales.
>
> L'idée de l'apprentissage par transfert est simple mais puissante: prendre un modèle déjà entraîné sur un grand jeu de données, comme ImageNet qui contient 1.2 million d'images de 1000 catégories, et le réutiliser pour notre tâche.
>
> Les premières couches du réseau ont déjà appris à reconnaître des caractéristiques générales comme les bords, les textures, les formes. Ces connaissances sont utiles pour n'importe quelle tâche d'image, y compris l'analyse médicale.
>
> On garde donc ces couches 'gelées' - on ne les modifie pas - et on remplace seulement la dernière couche pour l'adapter à notre problème de classification binaire: Normal ou Pneumonie.
>
> Cette technique nous permet d'obtenir d'excellents résultats avec seulement quelques milliers d'images au lieu de millions."

---

## Slide 7: Architecture du Modèle

**À LIRE:**

> "Voici l'architecture exacte de notre modèle:
>
> Nous partons de ResNet18 pré-entraîné. Les premières couches restent gelées - elles gardent leurs poids d'ImageNet.
>
> Nous remplaçons la dernière couche, appelée 'fully connected' ou FC, par notre propre tête de classification. Voici ce qu'elle contient:
>
> **Dropout 50%**: Pendant l'entraînement, on désactive aléatoirement la moitié des neurones. Cela force le réseau à ne pas trop se fier à des neurones spécifiques et rend le modèle plus robuste.
>
> **Couche linéaire 512→256**: On passe de 512 caractéristiques à 256. C'est une réduction de dimensionnalité.
>
> **Fonction ReLU**: L'activation ReLU, qui signifie Rectified Linear Unit, garde les valeurs positives et met les négatives à zéro. Cela ajoute de la non-linéarité au réseau.
>
> **Dropout 30%**: Une autre couche de régularisation, mais moins agressive.
>
> **Couche linéaire 256→1**: On réduit à une seule valeur.
>
> **Fonction Sigmoid**: Elle convertit cette valeur en une probabilité entre 0 et 1. Proche de 0 signifie Normal, proche de 1 signifie Pneumonie."

---

# 📖 PARTIE 3: JEU DE DONNÉES (3-4 minutes)

## Slide 8: Description du Jeu de Données

**À LIRE:**

> "Pour entraîner notre modèle, nous avons utilisé le jeu de données 'Chest X-Ray Images' disponible sur Kaggle.
>
> Ce jeu de données contient des radiographies thoraciques d'enfants de 1 à 5 ans, collectées au Guangzhou Women and Children's Medical Center en Chine.
>
> Il est divisé en trois ensembles:
>
> **L'ensemble d'entraînement**: 5,216 images utilisées pour apprendre les patterns. Il contient 1,341 images normales et 3,875 images de pneumonie.
>
> **L'ensemble de validation**: 16 images pour ajuster les paramètres pendant l'entraînement.
>
> **L'ensemble de test**: 624 images pour évaluer la performance finale. Il contient 234 images normales et 390 images de pneumonie.
>
> Vous remarquerez que le jeu de données est déséquilibré: il y a presque 3 fois plus de cas de pneumonie que de cas normaux. Nous avons dû en tenir compte dans notre entraînement."

---

## Slide 9: Prépaitement des Images

**À LIRE:**

> "Avant d'envoyer les images au réseau, nous devons les préparer. Cette étape s'appelle le prétraitement.
>
> **Redimensionnement**: Toutes les images sont redimensionnées à 224×224 pixels, car c'est la taille attendue par ResNet.
>
> **Normalisation**: Les valeurs des pixels sont normalisées en utilisant les moyennes et écarts-types d'ImageNet. Cela standardise les données et facilite l'apprentissage.
>
> Pour l'entraînement, nous ajoutons aussi de l'augmentation de données:
>
> **Retournement horizontal**: L'image peut être inversée comme un miroir.
>
> **Rotation**: L'image peut être tournée jusqu'à 15 degrés.
>
> **Translation**: L'image peut être légèrement décalée.
>
> **Variation de couleur**: La luminosité et le contraste peuvent varier légèrement.
>
> Ces augmentations créent artificiellement plus de variété dans les données, ce qui aide le modèle à mieux généraliser et évite le surapprentissage."

---

# 📖 PARTIE 4: ENTRAÎNEMENT (4-5 minutes)

## Slide 10: Processus d'Entraînement

**À LIRE:**

> "Maintenant, parlons de comment le modèle apprend.
>
> L'entraînement se fait en plusieurs 'epochs'. Une epoch, c'est quand le modèle a vu toutes les images d'entraînement une fois. Nous avons entraîné pendant 15 epochs.
>
> À chaque epoch, les images sont divisées en 'batches' de 32 images. Pour chaque batch:
>
> **Étape 1 - Passage avant**: Les images traversent le réseau et produisent des prédictions.
>
> **Étape 2 - Calcul de l'erreur**: On compare les prédictions aux vraies étiquettes avec la fonction de perte Binary Cross Entropy. Plus l'erreur est grande, plus le modèle s'est trompé.
>
> **Étape 3 - Rétropropagation**: L'erreur se propage à rebours dans le réseau pour calculer comment chaque poids a contribué à l'erreur.
>
> **Étape 4 - Mise à jour**: Les poids sont ajustés pour réduire l'erreur. Nous utilisons l'optimiseur Adam avec un taux d'apprentissage de 0.001.
>
> Ce processus se répète pour chaque batch, puis pour chaque epoch, jusqu'à ce que le modèle converge vers de bonnes performances."

---

## Slide 11: Hyperparamètres

**À LIRE:**

> "Les hyperparamètres sont les réglages que l'on choisit avant l'entraînement. Voici ceux que nous avons utilisés:
>
> **Taille des images**: 224×224 pixels, standard pour ResNet.
>
> **Taille du batch**: 32 images à la fois. C'est un bon compromis entre vitesse et stabilité.
>
> **Nombre d'epochs**: 15 passages sur les données.
>
> **Taux d'apprentissage initial**: 0.001, qui diminue automatiquement si le modèle stagne.
>
> **Dropout**: 50% et 30% pour la régularisation.
>
> Nous avons aussi utilisé un planificateur de taux d'apprentissage qui réduit le taux par 5 si la perte de validation ne s'améliore pas pendant 3 epochs. Cela permet un réglage fin du modèle vers la fin de l'entraînement."

---

# 📖 PARTIE 5: RÉSULTATS (3-4 minutes)

## Slide 12: Métriques de Performance

**À LIRE:**

> "Voici les résultats de notre modèle sur l'ensemble de test:
>
> **Précision globale (Accuracy)**: 89.26%. Cela signifie que sur 100 images, le modèle en classe correctement 89.
>
> **Précision (Precision)**: 86.62%. Parmi toutes les images que le modèle prédit comme pneumonie, 86.62% le sont vraiment.
>
> **Rappel (Recall)**: 97.95%. C'est notre métrique la plus importante! Sur 100 vrais cas de pneumonie, le modèle en détecte 98. Nous ne manquons presque aucun cas malade.
>
> **Score F1**: 91.94%. C'est la moyenne harmonique de la précision et du rappel, donnant une vue équilibrée.
>
> **AUC**: 0.9683. L'aire sous la courbe ROC est proche de 1, indiquant une excellente capacité de discrimination.
>
> Le rappel élevé est crucial en médecine: il vaut mieux avoir quelques faux positifs que de manquer de vrais cas de pneumonie."

---

## Slide 13: Matrice de Confusion

**À LIRE:**

> "La matrice de confusion nous montre exactement où le modèle se trompe.
>
> Sur les 234 images normales:
> - 175 ont été correctement classées comme normales (vrais négatifs)
> - 59 ont été incorrectement classées comme pneumonie (faux positifs)
>
> Sur les 390 images de pneumonie:
> - 382 ont été correctement détectées (vrais positifs)
> - Seulement 8 ont été manquées (faux négatifs)
>
> Ces 8 cas manqués représentent seulement 2% des pneumonies. C'est un excellent résultat pour une application médicale."

---

# 📖 PARTIE 6: APPLICATION WEB (3-4 minutes)

## Slide 14: Interface Streamlit

**À LIRE:**

> "Pour rendre notre modèle accessible, j'ai créé une application web avec Streamlit.
>
> Streamlit est un framework Python qui permet de créer des applications web interactives très facilement. En quelques lignes de code, on peut créer une interface complète.
>
> L'application se compose de:
>
> **Une zone de téléchargement**: L'utilisateur peut glisser-déposer ou sélectionner une radiographie.
>
> **L'affichage de l'image**: L'image téléchargée est affichée pour confirmation.
>
> **Les résultats**: Le diagnostic (Normal ou Pneumonie) s'affiche avec un code couleur - vert pour normal, rouge pour pneumonie.
>
> **Le score de confiance**: Un pourcentage indique à quel point le modèle est certain de sa prédiction.
>
> **La barre latérale**: Elle affiche les performances du modèle et un avertissement médical."

---

## Slide 15: Fonctionnement de l'Application

**À LIRE:**

> "Voici ce qui se passe quand vous utilisez l'application:
>
> **Étape 1**: Au démarrage, l'application charge le modèle entraîné en mémoire. Grâce au cache de Streamlit, cela ne se fait qu'une seule fois.
>
> **Étape 2**: Quand vous téléchargez une image, elle est convertie en format RGB si nécessaire.
>
> **Étape 3**: L'image est redimensionnée à 224×224 pixels et normalisée exactement comme pendant l'entraînement.
>
> **Étape 4**: L'image préparée passe dans le réseau de neurones qui produit une probabilité entre 0 et 1.
>
> **Étape 5**: Si la probabilité est supérieure à 0.5, le diagnostic est 'Pneumonie', sinon c'est 'Normal'.
>
> **Étape 6**: Les résultats s'affichent instantanément avec le niveau de confiance."

---

# 📖 PARTIE 7: DÉMONSTRATION LIVE (2-3 minutes)

## Slide 16: Démo

**À LIRE:**

> "Permettez-moi de vous faire une démonstration en direct de l'application.
>
> [Ouvrir l'application sur l'écran]
>
> Comme vous pouvez le voir, l'interface est simple et intuitive. Testons avec quelques images..."

**[FAIRE LA DÉMO LIVE]**

---

# 📖 PARTIE 8: CONCLUSION (2 minutes)

## Slide 17: Résumé et Perspectives

**À LIRE:**

> "Pour conclure, ce projet démontre comment l'intelligence artificielle peut assister le diagnostic médical.
>
> **Ce que nous avons accompli**:
> - Un modèle atteignant 89.26% de précision
> - Un rappel de 97.95%, minimisant les cas manqués
> - Une application web accessible et facile à utiliser
>
> **Limites actuelles**:
> - Le modèle ne distingue pas entre pneumonie bactérienne et virale
> - Il a été entraîné uniquement sur des radiographies d'enfants
> - Ce n'est pas un outil de diagnostic officiel
>
> **Améliorations futures possibles**:
> - Classification multi-classes pour différents types de pneumonie
> - Visualisation des zones suspectes avec Grad-CAM
> - Entraînement sur un jeu de données plus diversifié
> - Déploiement sur mobile pour les zones rurales
>
> Je vous remercie pour votre attention. Avez-vous des questions?"

---

# 📚 ANNEXE: RÉPONSES AUX QUESTIONS POTENTIELLES

## Q: Pourquoi utiliser PyTorch plutôt que TensorFlow?

> "PyTorch et TensorFlow sont les deux frameworks d'apprentissage profond les plus populaires. J'ai choisi PyTorch pour sa flexibilité et son approche plus 'pythonique'. Il est aussi très populaire dans la recherche académique."

## Q: Comment gérez-vous le déséquilibre des classes?

> "Nous utilisons des poids de classe automatiques et l'augmentation de données. Le fait que nous optimisons pour le rappel plutôt que la précision aide aussi à ne pas sous-détecter la classe minoritaire."

## Q: Le modèle peut-il fonctionner sur un téléphone?

> "Actuellement non, car PyTorch est assez lourd. Mais on pourrait convertir le modèle en TensorFlow Lite ou ONNX pour le déployer sur mobile."

## Q: Quelle est la différence entre précision et rappel?

> "La précision dit: parmi mes prédictions positives, combien sont correctes? Le rappel dit: parmi tous les vrais positifs, combien ai-je détectés? En médecine, le rappel est crucial car on veut détecter tous les malades."

## Q: Combien de temps a pris l'entraînement?

> "Sur un GPU NVIDIA, environ 30-45 minutes. Sur CPU, cela peut prendre plusieurs heures."

---

**FIN DU SCRIPT**

**Temps total estimé: 25-30 minutes de présentation**

---

## LIENS UTILES

- **GitHub:** https://github.com/samidardar/pneumonia-detection-CNN
- **Jeu de données:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Déploiement Streamlit:** https://share.streamlit.io
