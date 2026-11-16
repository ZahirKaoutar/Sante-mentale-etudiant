
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import display

def load_data(df):

    shape = df.shape
    null_colonne = df.isnull().sum()
    duplicated = df.duplicated().sum()
    describ = df.describe().T
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    print("========== DATA INFO ==========")
    df.info()

    print("\n========== SHAPE ==========")
    print(shape)

    print("\n========== MISSING VALUES ==========")
    display(null_colonne)

    print("\n========== MISSING PERCENTAGE ==========")
    display(missing_percentage.round(2))

    print("\n========== DUPLICATED ==========")
    print(duplicated)

    print("\n========== DESCRIPTIVE STATISTICS ==========")
    display(describ)


def anormal_logic(df):

    ilogis_time = df[df["Social_Media_Hours"] > df["Screen_Time_Hours"]]

    reelle_age_study = df[df["Age"] == 17][["Year_of_Study","Age"]]

    ilogic_year_study = df[
        (df["Age"] <= 18) & (df["Year_of_Study"] > 3)
    ]

    print("\n========== SOCIAL MEDIA / SCREEN TIME ==========")
    display(ilogis_time)

    print("\n========== STUDY YEAR FOR STUDENTS  ==========")
    display(reelle_age_study)

    print("\n========== AGE / YEAR INCONSISTENCIES ==========")
    display("ilogic_year_study",ilogic_year_study)
    
    
numeric_cols = ["Age", "Sleep_Hours", "Study_Hours_Per_Week", "GPA", "Year_of_Study","Social_Media_Hours","Screen_Time_Hours","Financial_Stress","Family_Support","Attendance_Rate"]
categorical_cols = ["Gender", "Physical_Activity","Previous_Mental_Health_History","Diet_Quality", "Part_Time_Job"]
target = "Mental_Health_Risk"
risk_order = ["Low", "Medium", "High"]

def plot_destribution(col, df):

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogramme
    sns.histplot(data=df, x=col, kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution de {col}")

    # Boxplot
    sns.boxplot(data=df, x=col, ax=axes[1])
    axes[1].set_title(f"Boxplot de {col}")

    plt.tight_layout()
    plt.show()

    # Calcul des outliers
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print("Column:", col)
    print("Mean:", round(df[col].mean(), 2))
    print("Median:", round(df[col].median(), 2))
    print("Skewness:", round(df[col].skew(), 2))
    print("Min:", df[col].min())
    print("Max:", df[col].max())
    print("Missing:", df[col].isnull().sum())
    print("Missing:", round(df[col].isnull().mean() * 100, 2), "%")
    print("Lower bound:", round(lower, 2))
    print("Upper bound:", round(upper, 2))
    print("Below lower bound:", len(df[df[col] < lower]))
    print("Above upper bound:", len(df[df[col] > upper]))
    print("Total outliers:", len(outliers))

    print("-" * 40)
    

    
    
    
def analyse_categorie(col,df):
    
    plt.figure(figsize=(7, 3.5))
    sns.countplot(data=df, x=col, palette="Set2", hue=col, legend=False)
    plt.title(f"Distribution de {col}", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    count_null=df[col].isnull().sum()
    pourcentage=(df[col].isnull().sum()/len(df))*100
    categori=df[col].nunique()
    count_categorie=df[col].value_counts()
    print("col",col)
    print("count_missgn colonne:",count_null)
    print("pourcentage de missign:",pourcentage)
    print("number_categori:",categori)
    print("count_categorie:",count_categorie)




def categorical_vs_target(df, target, c):
    plt.figure(figsize=(9, 4))

    sns.countplot(
        data=df,
        x=c,
        hue=target
    )
    plt.title(f"Répartition de {c} selon le niveau de risque", fontsize=12, fontweight="bold")
    plt.xlabel(c)
    plt.ylabel("Nombre d'observations")
    plt.legend(title="Niveau de risque")
    plt.tight_layout()
    plt.show()
    
def heatmap(df):
    eda = df.copy()
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    eda["Risk_Num"] = eda["Mental_Health_Risk"].map(risk_map)
    corr = eda[numeric_cols+["Risk_Num"]].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation between numeric features")
    plt.show()
    
    #calculation de correcltion
    
    target_corr = eda[numeric_cols + ["Risk_Num"]].corr()["Risk_Num"]
    target_corr = target_corr.drop("Risk_Num")
    target_corr = target_corr.sort_values()
    print(target_corr)
    





   
    
    
   
    
