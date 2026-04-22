import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def to_dense(X):
    """Converts sparse matrix to dense array if needed."""
    if hasattr(X, 'toarray'):
        return X.toarray()
    return np.array(X)


def needs_dense(model):
    """Returns True if this model requires a dense array input."""
    return any(k in model.__class__.__name__
               for k in ['SVC', 'MLP', 'XGB', 'Gradient', 'Stacking'])


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Runs the model on test data and prints all key metrics.
    Automatically handles sparse vs dense input.
    """
    X_input = to_dense(X_test) if needs_dense(model) else X_test
    y_pred  = model.predict(X_input)

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
    """Draws and saves a confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    classes   = ["ADHD Negative", "ADHD Positive"]
    ticks     = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=11)
    ax.set_yticks(ticks); ax.set_yticklabels(classes, fontsize=11)
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
    fname = f'../models/{model_name.replace(" ", "_")}_cm.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
#  BEFORE / AFTER FEATURE SELECTION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_before_after_selection(models_dict,
                                   X_train_full, X_test_full,
                                   X_train_sel,  X_test_sel,
                                   y_train, y_test):
    """
    Trains every model on the full feature set (124 features) AND
    the selected feature set (20 features), then prints a side-by-side
    comparison table and saves a grouped bar chart.

    models_dict: {"Model Name": unfitted_sklearn_estimator, ...}
    """
    import copy

    results = []

    for name, model in models_dict.items():
        # ── BEFORE selection ──────────────────────────────────────────────
        m_before = copy.deepcopy(model)
        Xtr_b = to_dense(X_train_full) if needs_dense(model) else X_train_full
        Xte_b = to_dense(X_test_full)  if needs_dense(model) else X_test_full
        m_before.fit(Xtr_b, y_train)
        yp_b = m_before.predict(Xte_b)

        # ── AFTER selection ───────────────────────────────────────────────
        m_after = copy.deepcopy(model)
        Xtr_a = to_dense(X_train_sel) if needs_dense(model) else X_train_sel
        Xte_a = to_dense(X_test_sel)  if needs_dense(model) else X_test_sel
        m_after.fit(Xtr_a, y_train)
        yp_a = m_after.predict(Xte_a)

        results.append({
            'name'    : name,
            'b_acc'   : accuracy_score(y_test, yp_b)  * 100,
            'b_f1'    : f1_score(y_test, yp_b)        * 100,
            'b_rec'   : recall_score(y_test, yp_b)    * 100,
            'b_prec'  : precision_score(y_test, yp_b) * 100,
            'a_acc'   : accuracy_score(y_test, yp_a)  * 100,
            'a_f1'    : f1_score(y_test, yp_a)        * 100,
            'a_rec'   : recall_score(y_test, yp_a)    * 100,
            'a_prec'  : precision_score(y_test, yp_a) * 100,
        })

    # ── Print table ───────────────────────────────────────────────────────
    w = 22
    print(f"\n{'='*95}")
    print(f"  BEFORE vs AFTER FEATURE SELECTION  "
          f"(Before = 124 features,  After = 20 features)")
    print(f"{'='*95}")
    print(f"  {'Model':<{w}}  {'':>4}  "
          f"{'Accuracy':>9}  {'F1':>7}  {'Recall':>7}  {'Precision':>10}")
    print(f"  {'-'*88}")
    for r in results:
        print(f"  {r['name']:<{w}}  {'Before':>6}  "
              f"{r['b_acc']:>8.1f}%  {r['b_f1']:>6.1f}%  "
              f"{r['b_rec']:>6.1f}%  {r['b_prec']:>9.1f}%")
        diff_acc  = r['a_acc']  - r['b_acc']
        diff_f1   = r['a_f1']   - r['b_f1']
        diff_rec  = r['a_rec']  - r['b_rec']
        diff_prec = r['a_prec'] - r['b_prec']
        print(f"  {'':<{w}}  {'After':>6}  "
              f"{r['a_acc']:>8.1f}%  {r['a_f1']:>6.1f}%  "
              f"{r['a_rec']:>6.1f}%  {r['a_prec']:>9.1f}%")
        sign = lambda v: f"+{v:.1f}" if v >= 0 else f"{v:.1f}"
        print(f"  {'':<{w}}  {'Change':>6}  "
              f"{sign(diff_acc):>8}%  {sign(diff_f1):>6}%  "
              f"{sign(diff_rec):>6}%  {sign(diff_prec):>9}%")
        print(f"  {'-'*88}")
    print(f"{'='*95}")

    # ── Grouped bar chart ─────────────────────────────────────────────────
    names      = [r['name'] for r in results]
    b_accs     = [r['b_acc'] for r in results]
    a_accs     = [r['a_acc'] for r in results]
    b_f1s      = [r['b_f1']  for r in results]
    a_f1s      = [r['a_f1']  for r in results]

    x    = np.arange(len(names))
    w    = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Before vs After Feature Selection (124 → 20 features)',
                 fontsize=14, fontweight='bold', y=1.01)

    colors_b = '#5B8DB8'
    colors_a = '#2E75B6'

    for ax, b_vals, a_vals, metric in zip(
            axes, [b_accs, b_f1s], [a_accs, a_f1s], ['Accuracy (%)', 'F1 Score (%)']):
        bars_b = ax.bar(x - w/2, b_vals, w, label='Before (124 features)',
                        color=colors_b, edgecolor='white', linewidth=0.5)
        bars_a = ax.bar(x + w/2, a_vals, w, label='After (20 features)',
                        color=colors_a, edgecolor='white', linewidth=0.5)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_ylim(55, 95)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top','right']].set_visible(False)
        for bar in bars_b:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.4,
                    f'{bar.get_height():.1f}',
                    ha='center', va='bottom', fontsize=7.5, color='#444')
        for bar in bars_a:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.4,
                    f'{bar.get_height():.1f}',
                    ha='center', va='bottom', fontsize=7.5, color='#1a4f7a')

    plt.tight_layout()
    plt.savefig('../models/before_after_feature_selection.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: ../models/before_after_feature_selection.png")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  ROC + PR CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_and_pr_curves(models_dict, X_test, y_test):
    """
    Plots ROC and Precision-Recall curves for all models side by side
    on a single figure and saves to models/ folder.

    ROC curve  — shows tradeoff between catching real cases (TPR)
                 and raising false alarms (FPR).  AUC closer to 1 = better.

    PR curve   — shows tradeoff between precision and recall at every
                 threshold. More useful when positive class matters most.
                 AP (Average Precision) summarises the curve.
    """
    colors = {
        'Logistic Regression' : '#2E75B6',
        'SVM (Linear)'        : '#1D9E75',
        'XGBoost'             : '#E24B4A',
        'MLP'                 : '#EF9F27',
        'Stacking (LR+XGB)'   : '#534AB7',
    }
    default_colors = ['#2E75B6','#1D9E75','#E24B4A','#EF9F27','#534AB7','#D85A30']

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Comparison — ROC and Precision-Recall Curves',
                 fontsize=14, fontweight='bold')

    for idx, (name, model) in enumerate(models_dict.items()):
        X_input = to_dense(X_test) if needs_dense(model) else X_test
        color   = colors.get(name, default_colors[idx % len(default_colors)])

        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_input)[:, 1]
        else:
            y_scores = model.decision_function(X_input)

        # ── ROC ──────────────────────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc     = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name}  (AUC = {roc_auc:.3f})')

        # ── PR ───────────────────────────────────────────────────────────
        prec, rec, _ = precision_recall_curve(y_test, y_scores)
        ap           = average_precision_score(y_test, y_scores)
        ax_pr.plot(rec, prec, color=color, lw=2,
                   label=f'{name}  (AP = {ap:.3f})')

    # ── ROC formatting ────────────────────────────────────────────────────
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='Random  (AUC = 0.500)')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curves', fontsize=13)
    ax_roc.legend(loc='lower right', fontsize=9)
    ax_roc.grid(True, alpha=0.3, linestyle='--')
    ax_roc.spines[['top','right']].set_visible(False)
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.01])

    # ── PR formatting ─────────────────────────────────────────────────────
    # Baseline = fraction of positives in test set
    baseline = y_test.sum() / len(y_test)
    ax_pr.axhline(y=baseline, color='k', linestyle='--', lw=1,
                  label=f'Random  (AP = {baseline:.3f})')
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Precision-Recall Curves', fontsize=13)
    ax_pr.legend(loc='lower left', fontsize=9)
    ax_pr.grid(True, alpha=0.3, linestyle='--')
    ax_pr.spines[['top','right']].set_visible(False)
    ax_pr.set_xlim([-0.01, 1.01])
    ax_pr.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    plt.savefig('../models/roc_and_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: ../models/roc_and_pr_curves.png")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CURVE METRICS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'ROC-AUC':>9} {'PR-AP':>8}")
    print(f"  {'-'*44}")
    for name, model in models_dict.items():
        X_input = to_dense(X_test) if needs_dense(model) else X_test
        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_input)[:, 1]
        else:
            y_scores = model.decision_function(X_input)
        roc_auc = auc(*roc_curve(y_test, y_scores)[:2])
        ap      = average_precision_score(y_test, y_scores)
        print(f"  {name:<25} {roc_auc:>9.3f} {ap:>8.3f}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_all(models_dict, X_train, y_train, cv=5):
    """
    Runs 5-fold cross validation on all models.
    More reliable than a single train/test split.
    """
    print(f"\n{'='*55}")
    print(f"  5-FOLD CROSS VALIDATION")
    print(f"{'='*55}")
    print(f"  {'Model':<22} {'Mean F1':>10} {'Std Dev':>10}")
    print(f"  {'-'*44}")

    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models_dict.items():
        X_input = to_dense(X_train) if needs_dense(model) else X_train
        scores  = cross_val_score(model, X_input, y_train,
                                  cv=cv_strat, scoring='f1', n_jobs=-1)
        print(f"  {name:<22} {scores.mean()*100:>9.1f}% "
              f"{scores.std()*100:>9.1f}%")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(models_dict, X_test, y_test):
    """Prints final summary table sorted by accuracy."""
    print(f"\n{'='*65}")
    print(f"  FINAL MODEL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8} "
          f"{'Recall':>8} {'Precision':>10}")
    print(f"  {'-'*58}")

    results = []
    for name, model in models_dict.items():
        X_input   = to_dense(X_test) if needs_dense(model) else X_test
        y_pred    = model.predict(X_input)
        results.append((
            name,
            accuracy_score(y_test, y_pred) * 100,
            f1_score(y_test, y_pred)        * 100,
            recall_score(y_test, y_pred)    * 100,
            precision_score(y_test, y_pred) * 100,
        ))

    results.sort(key=lambda x: x[1], reverse=True)
    for name, acc, f1, rec, prec in results:
        print(f"  {name:<22} {acc:>9.1f}% {f1:>7.1f}% "
              f"{rec:>7.1f}% {prec:>9.1f}%")
    print(f"{'='*65}")
    print(f"\n  Best model: {results[0][0]} ({results[0][1]:.1f}% accuracy)")
