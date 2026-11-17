# 🧠 Prédiction du Risque de Santé Mentale chez les Étudiants

Projet complet de Machine Learning et MLOps pour prédire le niveau de risque de santé mentale (`Low`, `Medium`, `High`) des étudiants universitaires à partir de facteurs académiques, comportementaux, et psychosociaux.

---

## 📁 Architecture du Projet

```text
├── app/
│   └── app.py                     # Application web interactive Streamlit
├── dag/
│   └── mental_health_dag.py       # DAG Apache Airflow pour l'orchestration MLOps
├── data/
│   └── raw/
│       ├── student_mental_health.csv  # Dataset brut (2000 étudiants, 17 variables)
│       └── README.md                  # Dictionnaire des données et description
├── models/
│   └── model.joblib               # Modèle optimisé sérialisé
├── notebooks/
│   ├── edaa_test.ipynb            # 1. Analyse Exploratoire des Données (EDA)
│   ├── preprocessign.ipynb        # 2. Prétraitement et Pipelines scikit-learn
│   ├── train.ipynb                # 3. Entraînement et Benchmark des modèles
│   └── optimazer.ipynb            # 4. Optimisation des hyperparamètres (GridSearch)
├── src/
│   ├── __init__.py
│   ├── eda.py                     # Fonctions d'exploration et visualisations
│   ├── preprocessing.py           # Nettoyage, imputation, split et ColumnTransformer
│   ├── train.py                   # Définition des modèles, benchmark et évaluation
│   └── tune.py                    # Recherche d'hyperparamètres et évaluation finale
└── requirements.txt               # Dépendances du projet
```

---

## 🚀 Installation & Utilisation

### 1. Cloner et installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancer l'application interactive Streamlit
- **En local** :
  ```bash
  streamlit run app/app.py
  ```
- **Avec Docker (recommandé)** :
  ```bash
  docker compose up -d
  ```
  - **Streamlit App** : [http://localhost:8502](http://localhost:8502)
  - **Airflow Web UI** : [http://localhost:8088](http://localhost:8088) (Identifiants : `admin` / `admin`)

### 3. Exécuter le pipeline DAG (Airflow ou Standalone)
- **Mode Standalone (direct)** :
  ```bash
  python dag/mental_health_dag.py
  ```
- **Mode Airflow** :
  Déposer le fichier `dag/mental_health_dag.py` dans votre dossier `$AIRFLOW_HOME/dags/`.

---

## 📊 Résumé des Résultats

| Modèle | Accuracy CV | F1-Score Macro |
|---|---|---|
| **Logistic Regression (Optimisé)** | **~77.0%** | **0.77** |
| Gradient Boosting | 70.3% | 0.70 |
| Random Forest | 68.8% | 0.68 |
| Decision Tree | 55.4% | 0.55 |

- **Facteurs déterminants majeurs** : Stress financier, déficit de sommeil, antécédents médicaux, et temps d'écran excessif.
- **Facteurs protecteurs** : Sommeil suffisant, fort soutien familial, activité physique régulière et alimentation équilibrée.
