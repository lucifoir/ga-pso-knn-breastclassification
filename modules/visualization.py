"""
visualization.py
----------------
Visualizes distributions, correlations, and feature selections.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_label_distribution(y):
    plt.figure(figsize=(5,4))
    # sns.countplot(x=y.map({1:'Malignant', 0:'Benign'}), palette='pastel')
    sns.countplot(x=y.map({1:'Malignant', 0:'Benign'}), hue=y.map({1:'Malignant', 0:'Benign'}), palette='pastel', legend=False)
    plt.title("Label Distribution (Malignant vs Benign)")
    plt.ylabel("Count")
    plt.show()

def plot_heatmap(X, y):
    # df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
    df = X.copy()
    df["Label"] = y.values
    corr = df.corr()

    plt.figure(figsize=(14,8))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_feature_distributions(X, y, features):
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
    df["Label"] = y.values
    plt.figure(figsize=(14,6))
    for i, f in enumerate(features, 1):
        plt.subplot(1, len(features), i)
        # sns.boxplot(x="Label", y=f, data=df, palette="pastel")
        sns.boxplot(x="Label", y=f, data=df, hue="Label", palette="pastel", legend=False)
        plt.title(f)
    plt.tight_layout()
    plt.show()

def visualize_selected_features(X, y, feature_mask, method_name):
    features = [f"Feature_{i+1}" for i, v in enumerate(feature_mask) if v]
    df = pd.DataFrame(X[:, feature_mask], columns=features)
    df["Label"] = y.values

    plt.figure(figsize=(10,4))
    for i, f in enumerate(features[:4], 1):
        plt.subplot(1, min(4, len(features)), i)
        sns.boxplot(x="Label", y=f, data=df, palette="pastel")
        plt.title(f"Boxplot {f}")
    plt.suptitle(f"Selected Features ({method_name})")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title(f"Feature Correlation ({method_name})")
    plt.show()
