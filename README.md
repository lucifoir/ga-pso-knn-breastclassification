# GA–PSO Swarm Optimization for Breast Cancer Classification

This project applies **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)** to improve feature selection and hyperparameter tuning of a **K-Nearest Neighbors (KNN)** classifier using the **Breast Cancer Wisconsin (Diagnostic)** dataset.

---

## Best Model
**PSO + KNN** — Test Accuracy: **≈ 97 %**, F1-Score: **≈ 0.96**

---

## Overview
Using evolutionary and swarm-based optimization, this notebook demonstrates:

- Automated dataset download from **KaggleHub**
- Data preprocessing and scaling
- Feature selection using Genetic Algorithm (GA)
- Hyperparameter tuning using Particle Swarm Optimization (PSO)
- Hybrid GA + PSO optimization
- Evaluation and visualization of model performance

---

## Frameworks
Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · KaggleHub · Jupyter

---

## Dataset
[Breast Cancer Wisconsin (Diagnostic) — Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
Includes **30 numeric features** derived from digitized images of breast mass nuclei.  
Target labels:
- `1` — Malignant  
- `0` — Benign  

---

## Contents
- `main_ga_pso.ipynb` — main notebook  
- `modules/` — modularized optimization and visualization scripts  
- `requirements.txt` — dependencies  
- `README.md` — project overview  
- `.gitignore` — ignored files for clean version control

---

© 2025 — Developed using Python and Scikit-learn
