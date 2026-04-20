import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into 80% training and 20% testing.
    stratify=y ensures both splits have ~50% positive cases.
    random_state=42 means you get the same split every time you run it.
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


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.
    This is our baseline model — simple, fast, and interpretable.
    max_iter=1000 gives it enough steps to converge properly.
    """
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Logistic Regression trained successfully!")
    return model


def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest model.
    This is our stronger comparison model.
    n_estimators=200 means 200 decision trees are built and averaged.
    """
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1        # uses all CPU cores to train faster
    )
    model.fit(X_train, y_train)
    print("Random Forest trained successfully!")
    return model


def save_model(model, filename, folder='../models'):
    """
    Saves a trained model to disk so you never have to retrain from scratch.
    Uses pickle — the standard Python way to save objects.
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {filepath}")


def load_model(filename, folder='../models'):
    """
    Loads a previously saved model from disk.
    """
    filepath = os.path.join(folder, filename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from: {filepath}")
    return model