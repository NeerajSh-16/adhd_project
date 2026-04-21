import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — using GradientBoosting instead.")
    print("Install it with: pip install xgboost")


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into 80% training and 20% testing.
    stratify=y ensures both splits have ~50% positive cases.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Training set : {X_train.shape[0]} rows")
    print(f"Testing set  : {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def to_dense(X):
    """
    Converts sparse matrix to dense array if needed.
    SVM, MLP, and XGBoost require dense input.
    """
    if hasattr(X, 'toarray'):
        return X.toarray()
    return np.array(X)


def train_logistic_regression(X_train, y_train):
    """
    Baseline model — simple, fast, interpretable.
    Best accuracy on this dataset: 81.4%
    """
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Done!")
    return model


def train_svm(X_train, y_train):
    """
    Support Vector Machine with linear kernel.
    Finds the best boundary line between ADHD positive and negative.
    Linear kernel works best here — matches Logistic Regression at 81.4%.
    RBF kernel was tested and performed worse (75.5%) on this dataset.
    probability=True allows us to get confidence scores, not just yes/no.
    """
    print("\nTraining SVM (linear kernel)...")
    X_train_d = to_dense(X_train)
    model = SVC(
        kernel='linear',
        probability=True,
        random_state=42
    )
    model.fit(X_train_d, y_train)
    print("Done!")
    return model


def train_xgboost(X_train, y_train):
    """
    XGBoost — Gradient Boosting model.
    Builds 200 decision trees one after another,
    each one correcting the mistakes of the previous one.
    Best config found by testing: lr=0.05, 200 trees, depth=4.
    Achieved 79.4% accuracy with 80.4% recall.
    """
    print("\nTraining XGBoost...")
    X_train_d = to_dense(X_train)

    if XGBOOST_AVAILABLE:
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )

    model.fit(X_train_d, y_train)
    print("Done!")
    return model


def train_mlp(X_train, y_train):
    """
    MLP — Multi-Layer Perceptron (simple neural network).
    Architecture: 128 neurons -> 64 neurons -> 32 neurons -> output.
    early_stopping=True stops training when performance stops improving.
    Note: MLP performed weakest on this dataset (69.6%) because
    506 rows is too small for neural networks to generalise well.
    Included for completeness and comparison purposes.
    """
    print("\nTraining MLP (Neural Network)...")
    X_train_d = to_dense(X_train)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        alpha=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_d, y_train)
    print("Done!")
    return model


def train_stacking(X_train, y_train):
    """
    Stacking — combines predictions of multiple models.
    Base models: Logistic Regression + XGBoost (best two performers).
    Meta model: Logistic Regression decides how to combine base predictions.

    How stacking works:
    1. LR and XGBoost each make their own predictions
    2. Those predictions become NEW features
    3. A final Logistic Regression learns the best combination
    This gave the best result: 83.3% accuracy on this dataset.

    cv=5 means each base model is trained on 5 different subsets
    to avoid data leakage between base and meta model.
    """
    print("\nTraining Stacking Ensemble (LR + XGBoost)...")
    X_train_d = to_dense(X_train)

    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42,
            eval_metric='logloss', verbosity=0
        )
    else:
        xgb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        )

    base_models = [
        ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
        ('xgb', xgb)
    ]

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )
    model.fit(X_train_d, y_train)
    print("Done!")
    return model


def train_random_forest(X_train, y_train):
    """
    Random Forest — kept from previous version for comparison.
    """
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Done!")
    return model


def save_model(model, filename, folder='../models'):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {filepath}")


def load_model(filename, folder='../models'):
    filepath = os.path.join(folder, filename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded: {filepath}")
    return model