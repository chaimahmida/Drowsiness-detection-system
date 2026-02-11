# 🚗 Real-Time Drowsiness Detection System with Dash UI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Dash_Plotly-orange.svg)](https://dash.plotly.com/)
[![AI](https://img.shields.io/badge/AI-TensorFlow_/_OpenCV-white.svg)](https://tensorflow.org/)

Ce projet est une application web interactive permettant de détecter en temps réel les signes de somnolence chez un conducteur. Il utilise un réseau de neurones convolutifs (CNN) pour classifier l'état des yeux et une interface moderne pour le monitoring.

---

## 🌟 Points Forts du Projet

* **Interface Web Moderne :** Développée avec `Dash` et `Bootstrap` pour une expérience utilisateur fluide.
* **Architecture Multithread :** Séparation du flux vidéo (OpenCV) et de l'interface (Dash) pour éviter tout ralentissement (lag) lors de l'inférence du modèle.
* **Système de Score Intelligent :** L'alerte ne se déclenche pas au moindre clignement, mais via un algorithme de cumul de score basé sur la persistance.
* **Alerte Sonore & Visuelle :** Intégration de `pygame.mixer` pour une alerte sonore immédiate et changement dynamique de l'interface.

---

## 🛠️ Stack Technique

* **Langage :** Python 3.x
* **Deep Learning :** TensorFlow / Keras (Modèle CNN personnalisé)
* **Vision par Ordinateur :** OpenCV (Haar Cascades pour la détection de visage/yeux)
* **Interface Utilisateur :** Dash (Plotly), Dash Bootstrap Components
* **Gestion Multitâche :** Threading & Locks (concurrence sécurisée)
* **Audio :** Pygame

---

## 💡 Fonctionnement Logique

Le système suit un pipeline de traitement précis :
1. **Capture :** Un thread dédié récupère les images de la webcam.
2. **Détection :** Utilisation de Haar Cascades pour isoler les régions d'intérêt (ROI) : le visage et les yeux.
3. **Prétraitement :** Les images des yeux sont converties en niveaux de gris et redimensionnées en $24 \times 24$ pixels.
4. **Inférence :** Le modèle CNN prédit si l'œil est `Ouvert` ou `Fermé`.
5. **Décision :** - Si les deux yeux sont fermés : `Score +1`
   - Si les yeux sont ouverts : `Score -1` (minimum 0)
   - Si `Score > 10` : Déclenchement de l'alarme.



[Image of Convolutional Neural Network architecture for image classification]
## 📥 Téléchargement des ressources (Modèles et Data)
* **Dataset :** [(https://drive.google.com/drive/folders/1CAZ5wcQ28jkaQyDFxYwy8bjVIHeliUbL?usp=sharing)]

---

## 📂 Structure du Répertoire

```text
├── assets/             # Fichiers audio (alarm.wav) et images
├── haarcascadefiles/   # Modèles XML pour la détection faciale
├── models/             # Modèle CNN entraîné (model.h5)
├── app.py              # Script principal (Dash Application)
├── requirements.txt    # Dépendances du projet
└── README.md           # Documentation
