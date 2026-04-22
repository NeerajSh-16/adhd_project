"""
One-time script to save the fitted TF-IDF vectorizer and SelectKBest selector
so that app.py can use them for live predictions.

Run once:  python save_artifacts.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pickle
from data_loader import load_data, create_target, handle_missing
from feature_engineering import (
    get_numeric_features, get_categorical_features,
    get_tfidf_features, apply_feature_selection,
    build_feature_matrix
)
from model import split_data
import scipy.sparse as sp

# ── 1. Load & clean data ─────────────────────────────────────────────────
df = load_data("data/ADHD.xlsx")
df = handle_missing(df)
df = create_target(df)
y  = df["target"]

# ── 2. Build features (this fits the vectorizer internally) ──────────────
numeric_df  = get_numeric_features(df)
category_df = get_categorical_features(df)
tfidf_matrix, vectorizer = get_tfidf_features(df)      # ← fitted vectorizer

numeric_sparse  = sp.csr_matrix(numeric_df.values.astype(float))
category_sparse = sp.csr_matrix(category_df.values.astype(float))
X = sp.hstack([numeric_sparse, category_sparse, tfidf_matrix])

print(f"Full feature matrix: {X.shape}")

# ── 3. Split data (same random_state as training) ────────────────────────
X_train, X_test, y_train, y_test = split_data(X, y)

# ── 4. Fit feature selector (same k=20 as training) ─────────────────────
X_train_sel, X_test_sel, selector = apply_feature_selection(
    X_train, y_train, X_test, k=20
)

# ── 5. Save artifacts ───────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("[OK] Saved: models/tfidf_vectorizer.pkl")

with open("models/feature_selector.pkl", "wb") as f:
    pickle.dump(selector, f)
print("[OK] Saved: models/feature_selector.pkl")

# ── 6. Also save the label encoder mappings for verification ─────────────
from sklearn.preprocessing import LabelEncoder
import pandas as pd

cat_cols = {
    'sex': 'sex',
    'diagnosed': 'have_you_ever_been_diagnosed_with_a_mental_illness',
    'on_medication': 'are_you_currently_using_prescribed_psychiatric_medication_for_a_mental_illness_or_symptoms_of_one',
    'prior_mh': 'have_you_ever_experienced_any_mental_health_difficulties_or_symptoms_before_starting_university_e_g_in_primary_or_high_school'
}
print("\n-- Label Encoder Mappings (verify these match app.py) --")
le = LabelEncoder()
for short_name, col_name in cat_cols.items():
    le.fit(df[col_name].astype(str))
    print(f"  {short_name}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\n[DONE] All artifacts saved! app.py should now work correctly.")
