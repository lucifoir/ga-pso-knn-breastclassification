"""
data_loader.py
---------------
Handles dataset acquisition from Kaggle and basic formatting.

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
"""

import kagglehub
import pandas as pd

def load_breast_cancer_data():
    """
    Downloads and loads the Breast Cancer Wisconsin dataset.
    Returns:
        X (DataFrame): Feature matrix (30 numeric features)
        y (Series): Label vector (0 = Benign, 1 = Malignant)
    """
    path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
    data = pd.read_csv(f"{path}/data.csv")

    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    return X, y
