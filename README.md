# 🩺 Détection de la Pneumonie avec Transfer Learning

Bienvenue dans ce notebook de démonstration qui illustre l’utilisation du **transfer learning** pour détecter la pneumonie à partir d’images médicales (scanners thoraciques).

---

## 📋 Description du projet

Nous exploitons le dataset **[Chest XRay Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** pour affiner un modèle pré-entraîné Keras et classer des images en présence ou non de pneumonie.

Le notebook suit un workflow clair et structuré, en 9 étapes :

| Étape | Description |
|-------|-------------|
| 1️⃣ | **Compréhension du problème** : analyser la problématique pour guider nos choix techniques et métriques. |
| 2️⃣ | **Analyse du dataset** : explorer les données, leur format, et la répartition des classes. |
| 3️⃣ | **Préprocessing** : adapter les images au format attendu par le modèle. |
| 4️⃣ | **Création du modèle** : construction via une fonction paramétrable. |
| 5️⃣ | **Entraînement** : apprentissage avec suivi des performances à chaque epoch. |
| 6️⃣ | **Suivi avec MLflow** : enregistrer paramètres et résultats pour comparaison. |
| 7️⃣ | **Sélection du meilleur modèle** : comparaison des modèles testés via MLflow. |
| 8️⃣ | **Optimisation** : ajustement des hyperparamètres pour maximiser les performances. |
| 9️⃣ | **Résultats finaux** : présentation du modèle final prêt à l’usage. |

---

## 🚀 Utilisation

Pour utiliser ce notebook, suivez les étapes ci-dessous :

```bash
# 1. Créer un environnement virtuel

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'interface MLflow
mlflow ui
```

L’interface MLflow sera accessible sur : http://localhost:5000

Ouvrez le notebook et exécutez les cellules dans l’ordre pour reproduire l’entraînement et l’évaluation.

---

Les images utilisées pour l'entraînement, la validation et le test proviennent du jeu de données suivant :

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Licence : CC BY 4.0

⚠️ Utilisation dans ce projet à but pédagogique uniquement.