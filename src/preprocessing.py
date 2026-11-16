"""Module de prétraitement (Preprocessing Pipeline) pour le projet Student Mental Health."""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Union, Any
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import numpy as np



def illogic_value(df):

    mask1 = df["Social_Media_Hours"] > df["Screen_Time_Hours"]
    # print(df["Social_Media_Hours"].isnull().sum())

    # mask2 = (df["Age"] <= 18) & (df["Year_of_Study"] > 3)

    # print("Social Media incohérent :", mask1.sum())
    

    df.loc[mask1, "Social_Media_Hours"] = np.nan
    # print(df["Social_Media_Hours"].isnull().sum())

    return df





def detecter_outlier(df,c):
    # #z-score:
    outliers=[]
    # z_score=(df[c]-df[c].mean())/df[c].std()
    # for i,v in z_score.items():
    #     if abs(v)>3:
    #         outliers.append({
    #             "col":c,
    #             "z-score":v
    #         })
      #iqr:
    q1=df[c].quantile(0.25)
    q3=df[c].quantile(0.75)
    iqr=q3-q1
    lower=q1-iqr*1.5
    upper= q3+1.5*iqr
    outl=df[(df[c]>upper) |(df[c]<lower)]
    if len(outl)>0:
        outliers.append({

                        "col":c,
                        "lower":lower,
                        "upper":upper,
                        "zone_normal":iqr,
                        "value_outlier":outl[c],
                        "len outlier":len(outl)
                        
                    })
    return outliers




def split_data(df):

    X = df.drop(columns=[
        "Student_ID",
        "Mental_Health_Risk",
        "Age"
    ])

    y = df["Mental_Health_Risk"].map({
        "Low": 0,
        "low": 0,
        "Medium": 1,
        "High": 2
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test



def preprocessing():

    numeric_cols = [
        "Year_of_Study",
        "Study_Hours_Per_Week",
        "Sleep_Hours",
        "Screen_Time_Hours",
        "Family_Support",
        "Financial_Stress",
        "Attendance_Rate",
        "GPA",
        "Social_Media_Hours"
    ]

    binary_cols = [
        "Part_Time_Job",
        "Previous_Mental_Health_History",
    ]

   

    ordinal_cols = [
        "Physical_Activity",
        "Diet_Quality"
    ]
    nominal_col=["Gender"]

    # Pipeline numérique
    pipeline_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="if_binary"))
])

    # Pipeline binaire et nominal
    pipeline_onehot = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())
    ])

    # Pipeline ordinal
    pipeline_ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["Low", "Moderate", "High"],
                ["Poor", "Average", "Good"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", pipeline_num, numeric_cols),
        ("binary", pipeline_onehot, binary_cols),
        ("nominale", nominal_pipeline,nominal_col ),
        ("ordinal", pipeline_ordinal, ordinal_cols)
    ])

    return preprocessor





# def preprocessign_data(df):
#     df=illogic_value(df.copy())
#     X_train, X_test, y_train, y_test = split_data(df)
#     preprocessor = preprocessing()
#     X_train_processed = preprocessor.fit_transform(X_train)
#     X_test_processed = preprocessor.transform(X_test)
#     print("Missing values Train:",
#               np.isnan(X_train_processed).sum())
    
#     print("Missing values Test:",
#               np.isnan(X_test_processed).sum())
    
    
#     return (
#         X_train_processed,
#         X_test_processed,
#         y_train,
#         y_test,
#         preprocessor
#     )




def preprocessign_data(df):
    df = illogic_value(df.copy())

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = preprocessing()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    
    
    
    

    



  
  
    

    








      
                
