import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Runs the model on test data and prints all key metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)

    print(f"\n{'='*45}")
    print(f"  Results for: {model_name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {accuracy*100:.1f}%")
    print(f"  F1 Score  : {f1*100:.1f}%")
    print(f"  Precision : {precision*100:.1f}%")
    print(f"  Recall    : {recall*100:.1f}%")
    print(f"{'='*45}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred,
          target_names=["ADHD Negative", "ADHD Positive"]))

    return y_pred


def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """
    Draws a confusion matrix — shows where the model is right and wrong.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes = ["ADHD Negative", "ADHD Positive"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11)

    # Write numbers inside each box
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('Actual label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, pad=12)

    plt.tight_layout()
    plt.savefig(f'../models/{model_name.replace(" ", "_")}_confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to models/ folder")


def plot_feature_importance(model, model_name="Model", top_n=20):
    """
    Shows the top N most important features.
    Works for Random Forest (uses feature_importances_).
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} does not support feature importance plot directly.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_labels = [f"Feature {i}" for i in indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(top_n), top_importances[::-1],
                   color='steelblue', edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels[::-1], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features — {model_name}',
                 fontsize=13, pad=12)

    plt.tight_layout()
    plt.savefig(f'../models/{model_name.replace(" ", "_")}_feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Feature importance plot saved to models/ folder")


def compare_models(models_dict, X_test, y_test):
    """
    Compares multiple models side by side in one summary table.
    models_dict = {"Logistic Regression": lr_model, "Random Forest": rf_model}
    """
    print(f"\n{'='*55}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>8} {'Recall':>8}")
    print(f"  {'-'*50}")

    for name, model in models_dict.items():
        y_pred    = model.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred) * 100
        f1        = f1_score(y_test, y_pred) * 100
        recall    = recall_score(y_test, y_pred) * 100
        print(f"  {name:<25} {accuracy:>9.1f}% {f1:>7.1f}% {recall:>7.1f}%")

    print(f"{'='*55}")