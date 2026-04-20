import pandas as pd
import os

def load_data(filepath):
    """
    Loads the ADHD Excel file and returns a clean dataframe.
    """
    # Check the file actually exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find the file at: {filepath}")

    # Load the main sheet
    df = pd.read_excel(filepath, sheet_name="Sheet1")
    print(f"Data loaded successfully — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_target(df):
    """
    Creates the binary target variable.
    ASRS Part A = items 1 to 6, summed.
    Score >= 14 means ADHD positive (1), else negative (0).
    """
    part_a_cols = [f'asrs1_item_{i}' for i in range(1, 7)]
    df['asrs_part_a'] = df[part_a_cols].sum(axis=1)
    df['target'] = (df['asrs_part_a'] >= 14).astype(int)

    pos = df['target'].sum()
    neg = len(df) - pos
    print(f"Target created — ADHD Positive: {pos}, Negative: {neg}")
    return df


def handle_missing(df):
    """
    Handles missing values for each column type.
    """
    # Free text diagnosis column — missing means no diagnosis
    text_col = 'if_you_have_been_diagnosed_formally_or_informally_please_list_the_diagnosis_diagnoses'
    df[text_col] = df[text_col].fillna('none')

    # Numeric columns — fill with column median
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    print("Missing values handled successfully")
    return df