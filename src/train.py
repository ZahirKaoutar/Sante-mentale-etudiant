import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

from src.preprocessing import (preprocessign_data,preprocessing)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report



def get_models():

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    return models


def benchmakig_model(df):

    models = get_models()

    data = preprocessign_data(df)

    X_train = data[0]
    y_train = data[2]
    preprocessor = data[4]

    results = []

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted"
    }

    for name, model in models.items():

        pipeline = Pipeline([
            ("processor", preprocessor),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring
        )

        results.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision Macro": scores["test_precision_macro"].mean(),
            "Recall Macro": scores["test_recall_macro"].mean(),
            "F1 Macro": scores["test_f1_macro"].mean(),
            "F1 Weighted": scores["test_f1_weighted"].mean()
        })

    benchmark = pd.DataFrame(results)

    benchmark = benchmark.sort_values(
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)

    return benchmark


def train_model(df): 


    benchmark = benchmakig_model(df)

    best_model_name = benchmark.iloc[0]["Model"]

    best_model = get_models()[best_model_name]



    data=preprocessign_data(df)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    preprocessor = data[4]



    pipline=Pipeline([
            ("preprocess",preprocessor),("model",best_model)
        ])
    pipline.fit(X_train,y_train)
    y_pred = pipline.predict(X_test)
    return(y_test,y_train,X_train,X_test,pipline,y_pred)
        
        
    
def affich_matrix(y_test,y_pred):
        cm = confusion_matrix(y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Low", "Medium", "High"]
            )
        
        disp.plot(cmap="Blues", values_format="d")
        plt.show()
        return disp

