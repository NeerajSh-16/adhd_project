import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import scipy.sparse as sp
from collections import Counter


def get_numeric_features(df):
    asrs_part_b = [f'asrs1_item_{i}' for i in range(7, 19)]
    bai_features = ['bai1_total', 'bai1_item_4', 'bai1_item_8']
    bdi_features = ['bdi1_total', 'bdi1_item_19']
    aas_features = ['aas1_item_3', 'aas1_item_4', 'aas1_item_6']
    all_numeric = asrs_part_b + bai_features + bdi_features + aas_features
    print(f"Numeric features selected: {len(all_numeric)}")
    return df[all_numeric]


def get_categorical_features(df):
    cat_cols = {
        'sex': 'sex',
        'diagnosed': 'have_you_ever_been_diagnosed_with_a_mental_illness',
        'on_medication': 'are_you_currently_using_prescribed_psychiatric_medication_for_a_mental_illness_or_symptoms_of_one',
        'prior_mh': 'have_you_ever_experienced_any_mental_health_difficulties_or_symptoms_before_starting_university_e_g_in_primary_or_high_school'
    }
    encoded = pd.DataFrame()
    le = LabelEncoder()
    for short_name, col_name in cat_cols.items():
        encoded[short_name] = le.fit_transform(df[col_name].astype(str))
    print(f"Categorical features encoded: {len(cat_cols)}")
    return encoded


def get_tfidf_features(df, max_features=100):
    text_col = 'if_you_have_been_diagnosed_formally_or_informally_please_list_the_diagnosis_diagnoses'
    text_data = df[text_col].astype(str).str.lower()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(text_data)
    print(f"TF-IDF features created: {tfidf_matrix.shape[1]} words/phrases")
    return tfidf_matrix, vectorizer


def check_class_imbalance(y):
    """
    Checks whether the target variable is imbalanced.
    Imbalanced means one class has many more samples than the other.
    Ratio > 1.5 is considered imbalanced.
    """
    counter = Counter(y)
    total = len(y)
    ratio = max(counter.values()) / min(counter.values())

    print("\n" + "="*45)
    print("  CLASS IMBALANCE CHECK")
    print("="*45)
    print(f"  ADHD Negative (0) : {counter[0]} ({counter[0]/total*100:.1f}%)")
    print(f"  ADHD Positive (1) : {counter[1]} ({counter[1]/total*100:.1f}%)")
    print(f"  Imbalance ratio   : {ratio:.3f}")

    if ratio > 1.5:
        print("  Status: IMBALANCED — handling required")
        print("="*45)
        return True
    else:
        print("  Status: BALANCED — no handling needed")
        print("="*45)
        return False


def apply_feature_selection(X_train, y_train, X_test, k=20):
    """
    Selects the k most useful features using Mutual Information.

    Mutual Information measures how much knowing a feature
    tells us about the target (ADHD positive/negative).
    Higher score = more useful feature.

    k=20 was chosen because testing on this dataset showed
    it gives the best accuracy (81.4%) — better than using
    all 124 features (79.4%).
    """
    print(f"\nApplying feature selection...")
    print(f"  Features before selection : {X_train.shape[1]}")

    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected  = selector.transform(X_test)

    print(f"  Features after selection  : {X_train_selected.shape[1]}")
    print(f"  Features removed          : {X_train.shape[1] - k}")
    return X_train_selected, X_test_selected, selector


def build_feature_matrix(df):
    """
    Combines numeric + categorical + TF-IDF into one matrix.
    """
    numeric_df             = get_numeric_features(df)
    category_df            = get_categorical_features(df)
    tfidf_matrix, vectorizer = get_tfidf_features(df)

    numeric_sparse  = sp.csr_matrix(numeric_df.values.astype(float))
    category_sparse = sp.csr_matrix(category_df.values.astype(float))

    X = sp.hstack([numeric_sparse, category_sparse, tfidf_matrix])
    print(f"\nFull feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    return X, vectorizer