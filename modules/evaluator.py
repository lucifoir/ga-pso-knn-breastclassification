"""
evaluator.py
------------
Provides model evaluation utilities (accuracy, precision, recall, F1-score)
for KNN models using selected features.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(X_train, y_train, X_test, y_test, feature_mask, k, metric, method):
    """
    Train and evaluate KNN using the given subset of features.
    Returns a dictionary of performance metrics.
    """
    X_train_sel = X_train[:, feature_mask]
    X_test_sel = X_test[:, feature_mask]

    model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n🔍 Evaluation — {method}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    return {
        "Method": method,
        "Accuracy": round(acc, 4),
        "Precision": round(pre, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4)
    }
