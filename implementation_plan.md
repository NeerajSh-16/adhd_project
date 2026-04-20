# Feature Selection & Class Imbalance — Implementation Plan

## 1. Class Imbalance Analysis

After inspecting the dataset, here are the findings:

| Metric | Value |
|---|---|
| Total samples | 506 |
| ADHD Positive (1) | 251 (49.6%) |
| ADHD Negative (0) | 255 (50.4%) |
| Class ratio | 0.984 |

> [!TIP]
> **The classes are almost perfectly balanced (≈50/50).** No resampling (SMOTE, undersampling) or class-weight adjustments are needed. This is great news — the models can learn both classes equally well without any special handling.

We will still add a **class imbalance check function** to `data_loader.py` so the pipeline documents this explicitly and would flag issues if the dataset ever changes.

---

## 2. Feature Selection — Why Mutual Information + SelectKBest?

Your current pipeline produces **~124 features** (20 numeric + 4 categorical + 100 TF-IDF) from only **506 samples**. That's a 1:4 feature-to-sample ratio, which risks overfitting — especially on the 100 sparse TF-IDF features.

I evaluated the main feature selection approaches for this use case:

| Method | Pros | Cons | Verdict |
|---|---|---|---|
| **Mutual Information + SelectKBest** | Works on mixed feature types (numeric, categorical, sparse TF-IDF); captures non-linear relationships; filter method so no risk of overfitting during selection | Slightly slower than chi² | ✅ **Best fit** |
| Chi-squared (χ²) | Fast, simple | Only works on non-negative features; misses non-linear patterns | ❌ Some features can be negative |
| LASSO (L1 Regularization) | Built-in to Logistic Regression | Doesn't work well with sparse TF-IDF; wrapper method can overfit on small data | ❌ Not ideal for mixed types |
| Recursive Feature Elimination (RFE) | Thorough | Very slow with 124 features; wrapper method overfits on small datasets | ❌ Too expensive |
| Tree-based importance | Good for tree models | Biased toward high-cardinality features; model-specific | ❌ Not generalizable |

> [!IMPORTANT]
> **Recommendation: Mutual Information with SelectKBest (top 40 features)**
> - Mutual Information measures how much knowing a feature reduces uncertainty about the target — it catches both linear AND non-linear relationships
> - SelectKBest is a filter method (independent of the model), so it won't introduce bias
> - Reducing from ~124 → 40 features cuts the noise from low-value TF-IDF words while keeping the strong clinical signals
> - The `k=40` default is configurable and can be tuned

---

## Proposed Changes

### Data Loader — Class Imbalance Check

#### [MODIFY] [data_loader.py](file:///c:/Users/Laptop/adhd_project/src/data_loader.py)
- Add `check_class_balance(y)` function that:
  - Prints class distribution counts and percentages
  - Calculates imbalance ratio
  - Warns if minority class < 30% (threshold for concern)
  - Returns a dict with balance statistics

---

### Feature Engineering — Feature Selection

#### [MODIFY] [feature_engineering.py](file:///c:/Users/Laptop/adhd_project/src/feature_engineering.py)
- Add `select_best_features(X, y, k=40)` function using `SelectKBest` with `mutual_info_classif`
- Add `get_feature_names(numeric_df, category_df, vectorizer)` helper to track feature names through selection
- Modify `build_feature_matrix()` to accept `y` and an optional `k` parameter
- The function returns the reduced feature matrix + the selector object (for applying to test data)

---

### Model Training — Apply Feature Selection

#### [MODIFY] [model.py](file:///c:/Users/Laptop/adhd_project/src/model.py)
- Update `split_data()` to return the selector alongside train/test splits
- Add `class_weight='balanced'` to both models as a safety net (no harm when classes are balanced, but protects against future dataset changes)

---

### Evaluation — Update Feature Importance

#### [MODIFY] [evaluate.py](file:///c:/Users/Laptop/adhd_project/src/evaluate.py)
- Update `plot_feature_importance()` to accept and display actual feature names instead of generic "Feature 0, Feature 1..."
- This will make the feature importance plot much more interpretable

---

## Verification Plan

### Automated Tests
- Run the full pipeline end-to-end and verify:
  - Class balance check prints correct stats
  - Feature selection reduces from ~124 to 40 features
  - Models train and evaluate successfully on the reduced feature set
  - Feature importance plot shows real feature names

### Manual Verification
- Compare model accuracy/F1 before and after feature selection to confirm no performance degradation
