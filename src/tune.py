
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.train import train_model

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


def search_params(df):

    data = train_model(df)

    y_test = data[0]
    y_train = data[1]
    X_train = data[2]
    X_test = data[3]
    pipeline = data[4]

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l1", "l2"],
        "model__solver": ["saga"],
        "model__class_weight": [None, "balanced"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Best accurcy macro:", grid_search.best_score_)
    print("Best Parameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    return best_model, X_test, y_test


def evaluate_final_model(df):

    best_model, X_test, y_test = search_params(df)

    # Predictions
    y_pred = best_model.predict(X_test)

    # Classification Report
    print("Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Medium", "High"]
    ))

    
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Medium", "High"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Final Model - Confusion Matrix")
    plt.show()

    return best_model, y_pred
