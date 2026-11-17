import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import joblib


# Ajouter la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data/raw/student_mental_health.csv"
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"


# TASK 1 : Vérifier les données
def check_data():
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset vide")

    print("Données vérifiées :", df.shape)

    return "Data validated"


# TASK 2 : Prétraitement
def preprocess_data():
    from src.preprocessing import preprocessign_data

    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor = preprocessign_data(df)

    print("Train :", X_train.shape)
    print("Test :", X_test.shape)

    return "Preprocessing completed"


# TASK 3 : Benchmark
def train_benchmark():
    from src.train import benchmakig_model

    df = pd.read_csv(DATA_PATH)

    results = benchmakig_model(df)

    best_model = results.iloc[0]["Model"]
    accuracy = results.iloc[0]["Accuracy"]

    print("Best model :", best_model)
    print("Accuracy :", accuracy)

    return "Benchmark completed"


# TASK 4 : Optimisation et sauvegarde
def tune_model():
    from src.tune import evaluate_final_model

    df = pd.read_csv(DATA_PATH)

    model, y_pred = evaluate_final_model(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("Model saved :", MODEL_PATH)

    return "Model saved successfully"


# AIRFLOW DAG
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    with DAG(
        dag_id="student_mental_health_pipeline",
        start_date=datetime(2026, 11, 14),
        schedule="@weekly",
        catchup=False,
        tags=["ML", "Mental Health"]
    ) as dag:

        task1 = PythonOperator(
            task_id="check_data",
            python_callable=check_data
        )

        task2 = PythonOperator(
            task_id="preprocess_data",
            python_callable=preprocess_data
        )

        task3 = PythonOperator(
            task_id="train_benchmark",
            python_callable=train_benchmark
        )

        task4 = PythonOperator(
            task_id="tune_model",
            python_callable=tune_model
        )

        task1 >> task2 >> task3 >> task4

except ImportError:
    dag = None


if __name__ == "__main__":
    check_data()
    preprocess_data()
    train_benchmark()
    tune_model()
