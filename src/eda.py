"""Module d'Analyse Exploratoire des Données (EDA) et Visualisation.

Ce module encapsule l'ensemble des étapes d'analyse exploratoire du projet
'Student Mental Health Risk' sous forme de fonctions réutilisables :
- Chargement et inspection générale
- Vérification des doublons et des valeurs manquantes
- Détection des incohérences logiques métier
- Statistiques descriptives et détection des outliers (IQR)
- Analyse et visualisation de la variable cible (Mental_Health_Risk)
- Relations entre variables explicatives et variable cible
- Analyse des corrélations (variables numériques uniquement)
- Classement des variables les plus fortement liées au niveau de risque
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Configuration par défaut
DEFAULT_ORDER = ["Low", "Medium", "High"]
DEFAULT_PALETTE = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
DEFAULT_NUM_COLS = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]
DEFAULT_CAT_COLS = [
    "Gender",
    "Physical_Activity",
    "Diet_Quality",
    "Part_Time_Job",
    "Previous_Mental_Health_History",
]



DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "student_mental_health.csv"

def load_and_inspect_dat(DEFAULT_DATA_PATH):
    """Charge le dataset des étudiants et affiche ses informations générales."""
    # path = Path(filepath) if filepath else DEFAULT_DATA_PATH
    # if not path.exists():
    #     # Fallback pour chemin relatif depuis notebooks/
    #     alt_path = Path("..") / "data" / "raw" / "student_mental_health.csv"
    #     path = alt_path if alt_path.exists() else path

    df = pd.read_csv(DEFAULT_DATA_PATH)
    print("--- Dimensions du jeu de données ---")
    print(f"Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}\n")
    print("--- Structure générale (df.info()) ---")
    df.info()
    return df


def check_duplicates_and_missing(df):
    
    nb_doublons = df.duplicated().sum()
    print(f"Nombre de doublons stricts : {nb_doublons}")
    if "Student_ID" in df.columns:
        print(f"Nombre d'étudiants uniques (Student_ID) : {df['Student_ID'].nunique()}")
    missing = df.isnull().sum()
    print("\nValeurs manquantes par colonne :")
    print(missing)

    missing_df = pd.DataFrame({
        "Valeurs_Manquantes": missing[missing > 0],
        "Pourcentage (%)": (missing[missing > 0] / len(df) * 100).round(2)
    }).sort_values(by="Valeurs_Manquantes", ascending=False)
    return missing_df
    
    
        

    
    

def check_logical_inconsistencies(df):

    incoherences = {}

    # 1. Réseaux sociaux > Temps d'écran total
    incoh_reseaux = df[df["Social_Media_Hours"] > df["Screen_Time_Hours"]]
    incoherences["social_gt_screen"] = len(incoh_reseaux)
    print(f"1. Incohérence Social_Media > Screen_Time : {len(incoh_reseaux)} cas")

    # 2. Âge <= 18 ans en année avancée (>= 3)
    incoh_age = df[(df["Age"] <= 18) & (df["Year_of_Study"] >= 3)]
    incoherences["jeune_en_annee_avancee"] = len(incoh_age)
    print(f"2. Incohérence Âge <= 18 ans en Année >= 3 : {len(incoh_age)} cas")

    # 3. Cumul journalier activités > 24h
    study_hour_day = df["Study_Hours_Per_Week"] / 7.0
    somme_activites_jour = df["Sleep_Hours"] + df["Screen_Time_Hours"] + study_hour_day
    incoh_24h = df[somme_activites_jour > 24.0]
    incoherences["activites_cumulees_gt_24h"] = len(incoh_24h)
    print(f"3. Incohérence Cumul journalier (Sommeil + Écran + Étude) > 24h : {len(incoh_24h)} cas")

    # 4. Sommeil + Écran > 24h
    incoh_sleep_screen = df[(df["Sleep_Hours"] + df["Screen_Time_Hours"]) > 24.0]
    incoherences["sommeil_plus_ecran_gt_24h"] = len(incoh_sleep_screen)
    print(f"4. Incohérence Sommeil + Écran > 24h : {len(incoh_sleep_screen)} cas")

    # 5. Sommeil + Étude journalière > 24h
    incoh_sleep_study = df[(df["Sleep_Hours"] + study_hour_day) > 24.0]
    incoherences["sommeil_plus_etude_gt_24h"] = len(incoh_sleep_study)
    print(f"5. Incohérence Sommeil + Étude/jour > 24h : {len(incoh_sleep_study)} cas")

    return incoherences




def get_descriptive_statistics(df):
    stats_df = df.describe().T.round(2)
    return stats_df



def detect_outliers(df):

    resultats = []

    # Sélectionner les colonnes numériques
    numeriques = df.select_dtypes(include="number")

    for col in numeriques.columns:

        # Calculer Q1, Q3 et IQR
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        # Calculer les limites
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Détecter les outliers
        outliers = df[(df[col] < lower) | (df[col] > upper)]

        # Enregistrer les résultats
        resultats.append({
            "variable": col,
            "nombre_outliers": len(outliers),
            "lower": round(lower, 2),
            "upper": round(upper, 2),
            "valeurs_outliers": outliers[col].dropna().tolist()
            
        })

    return pd.DataFrame(resultats)



def plot_target_distribution(
    df,
    target_col="Mental_Health_Risk",
    order=None,
    palette=None
):

   
    if order is None:
        order = DEFAULT_ORDER
    if palette is None:
        palette = DEFAULT_PALETTE

    plt.figure(figsize=(7, 4))
    sns.countplot(
        data=df,
        x=target_col,
        order=order,
        palette=palette,
        hue=target_col,
        legend=False
    )
    plt.title(f"Distribution de la Variable Cible : {target_col}", fontsize=13, fontweight="bold")
    plt.xlabel("Niveau de Risque")
    plt.ylabel("Nombre d'étudiants")
    plt.tight_layout()
    plt.show()

    print("--- Effectifs par classe : ---")
    print(df[target_col].value_counts())

    print("\n--- Pourcentages par classe (%) : ---")
    print((df[target_col].value_counts(normalize=True) * 100).round(2))

    print("\nConclusion : Les 3 classes sont équilibrées (~33% chacune). Aucun rééquilibrage SMOTE n'est nécessaire.")


def plot_feature_distributions(df):
    numerical_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    print("--- Affichage des distributions numériques ---")
    for col in numerical_cols:
        plt.figure(figsize=(7, 3.5))
        sns.histplot(data=df, x=col, kde=True, color="#3498db")
        plt.title(f"Distribution de {col}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.show()

    print("--- Affichage des distributions catégorielles ---")
    for col in categorical_cols:
        if col == "Student_ID":
            continue
        plt.figure(figsize=(7, 3.5))
        sns.countplot(data=df, x=col, palette="Set2", hue=col, legend=False)
        plt.title(f"Distribution de {col}", fontsize=11, fontweight="bold")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


# ============================================================================
# 4. RELATIONS VARIABLES EXPLICATIVES VS VARIABLE CIBLE
# ============================================================================

def plot_numerical_vs_target(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    target_col: str = "Mental_Health_Risk"
) -> None:
   
    if num_cols is None:
        num_cols = [c for c in DEFAULT_NUM_COLS if c in df.columns]

    for c in num_cols:
        plt.figure(figsize=(9, 4))
        sns.barplot(
            data=df,
            x=target_col,
            y=c,
            hue=target_col,
            order=DEFAULT_ORDER,
            palette=DEFAULT_PALETTE,
            errorbar=None
        )
        plt.title(f"Moyenne de {c} selon le risque", fontsize=12, fontweight="bold")
        plt.xlabel("Niveau de risque")
        plt.ylabel(f"Moyenne de {c}")
        plt.tight_layout()
        plt.show()


def plot_categorical_vs_target(
    df,
    categorical_cols = None,
    target_col= "Mental_Health_Risk",
    order= None
):
    
    if categorical_cols is None:
        categorical_cols = [c for c in DEFAULT_CAT_COLS if c in df.columns]
    if order is None:
        order = DEFAULT_ORDER

    colors = [DEFAULT_PALETTE[cat] for cat in order]

    for c in categorical_cols:
        proportions = pd.crosstab(
            df[c],
            df[target_col],
            normalize="index"
        )[order]

        proportions.plot(
            kind="bar",
            stacked=True,
            figsize=(9, 4.5),
            color=colors,
            edgecolor="white"
        )
        plt.title(f"Proportion du risque selon {c}", fontsize=12, fontweight="bold")
        plt.xlabel(c)
        plt.ylabel("Proportion")
        plt.legend(title="Mental Health Risk", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


# ============================================================================
# 5. CORRÉLATIONS (UNIQUEMENT VARIABLES NUMÉRIQUES)
# ============================================================================

def plot_numerical_correlation_matrix(
    df,
    method="spearman",
    target_col="Mental_Health_Risk",
    include_risk_score= True,
    figsize: Tuple[int, int] = (11, 7)
):
    
    df_corr = df.copy()

    num_features = list(df.select_dtypes(include=["float64", "int64"]).columns)

    if include_risk_score and target_col in df.columns:
        df_corr["Risk_Score"] = df[target_col].map({"Low": 0, "Medium": 1, "High": 2})
        features_to_corr = num_features + ["Risk_Score"]
    else:
        features_to_corr = num_features

    corr_matrix = df_corr[features_to_corr].corr(method=method)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )
    plt.title(
        f"Matrice de Corrélation de {method.capitalize()} (Variables Numériques Uniquement)",
        fontsize=13,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    return corr_matrix


# ============================================================================
# 6. VARIABLES LES PLUS FORTEMENT LIÉES AU NIVEAU DE RISQUE
# ============================================================================

def get_top_risk_correlations(
    df,
    method= "spearman",
    target_col= "Mental_Health_Risk"
):
  
    df_corr = df.copy()
    df_corr["Risk_Score"] = df[target_col].map({"Low": 0, "Medium": 1, "High": 2})

    num_features = df.select_dtypes(include=["float64", "int64"]).columns
    corrs = df_corr[num_features].apply(lambda x: df_corr["Risk_Score"].corr(x, method=method))
    ranking = corrs.abs().sort_values(ascending=False)

    print("--- Variables numériques classées par impact sur le risque : ---")
    for var in ranking.index:
        print(f"{var:25s} : r = {corrs[var]:+.3f}")

    return corrs.loc[ranking.index]

