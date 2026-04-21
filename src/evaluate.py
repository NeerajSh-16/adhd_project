import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def to_dense(X):
    if hasattr(X, 'toarray'):
        return X.toarray()
    return np.array(X)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Runs the model on test data and prints all key metrics.
    Automatically handles sparse vs dense input.
    """
    # SVM, MLP, XGBoost need dense arrays
    needs_dense = any(k in model.__class__.__name__
                      for k in ['SVC', 'MLP', 'XGB', 'Gradient',
                                'Stacking'])
    X_input = to_dense(X_test) if needs_dense else X_test

    y_pred = model.predict(X_input)

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
    print(classification_report(y_test, y_pred,
          target_names=["ADHD Negative", "ADHD Positive"]))

    return y_pred


def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """
    Draws a confusion matrix.
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
    plt.savefig(f'../models/{model_name.replace(" ", "_")}_cm.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plots ROC curves for all models on one chart.
    ROC curve shows the tradeoff between catching real cases
    and avoiding false alarms at different thresholds.
    AUC (Area Under Curve): closer to 1.0 = better model.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2E75B6', '#1D9E75', '#E24B4A',
              '#EF9F27', '#534AB7', '#D85A30']

    for (name, model), color in zip(models_dict.items(), colors):
        needs_dense = any(k in model.__class__.__name__
                          for k in ['SVC', 'MLP', 'XGB', 'Gradient',
                                    'Stacking'])
        X_input = to_dense(X_test) if needs_dense else X_test

        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_input)[:, 1]
        else:
            y_scores = model.decision_function(X_input)

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=13, pad=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../models/roc_curves_all_models.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("ROC curves saved to models/ folder")


def cross_validate_all(models_dict, X_train, y_train, cv=5):
    """
    Runs 5-fold cross validation on all models.
    More reliable than a single train/test split because
    it tests each model on 5 different subsets of the data.
    """
    print(f"\n{'='*55}")
    print(f"  5-FOLD CROSS VALIDATION")
    print(f"{'='*55}")
    print(f"  {'Model':<22} {'Mean F1':>10} {'Std Dev':>10}")
    print(f"  {'-'*44}")

    cv_strategy = StratifiedKFold(n_splits=cv,
                                  shuffle=True,
                                  random_state=42)

    for name, model in models_dict.items():
        needs_dense = any(k in model.__class__.__name__
                          for k in ['SVC', 'MLP', 'XGB',
                                    'Gradient', 'Stacking'])
        X_input = to_dense(X_train) if needs_dense else X_train

        scores = cross_val_score(model, X_input, y_train,
                                 cv=cv_strategy, scoring='f1',
                                 n_jobs=-1)
        print(f"  {name:<22} {scores.mean()*100:>9.1f}% "
              f"{scores.std()*100:>9.1f}%")

    print(f"{'='*55}")


def compare_models(models_dict, X_test, y_test):
    """
    Final summary table comparing all models.
    """
    print(f"\n{'='*65}")
    print(f"  FINAL MODEL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8} "
          f"{'Recall':>8} {'Precision':>10}")
    print(f"  {'-'*58}")

    results = []
    for name, model in models_dict.items():
        needs_dense = any(k in model.__class__.__name__
                          for k in ['SVC', 'MLP', 'XGB',
                                    'Gradient', 'Stacking'])
        X_input = to_dense(X_test) if needs_dense else X_test

        y_pred    = model.predict(X_input)
        accuracy  = accuracy_score(y_test, y_pred) * 100
        f1        = f1_score(y_test, y_pred) * 100
        recall    = recall_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred) * 100
        results.append((name, accuracy, f1, recall, precision))

    # Sort by accuracy descending
    results.sort(key=lambda x: x[1], reverse=True)

    for name, acc, f1, rec, prec in results:
        print(f"  {name:<22} {acc:>9.1f}% {f1:>7.1f}% "
              f"{rec:>7.1f}% {prec:>9.1f}%")

    print(f"{'='*65}")
    best = results[0]
    print(f"\n  Best model: {best[0]} ({best[1]:.1f}% accuracy)")