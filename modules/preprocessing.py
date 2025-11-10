"""
preprocessing.py
----------------
Splits and scales the dataset using StandardScaler.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Splits and scales the dataset.
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
