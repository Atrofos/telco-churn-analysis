"""Data cleaning for the Telco Customer Churn dataset.

Turns the raw CSV into a clean dataset ready for feature selection. Every
decision here traces back to a finding in notebooks/eda.ipynb.
"""

import os

import pandas as pd

RAW_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
CLEANED_PATH = os.path.join("data", "processed", "telco_cleaned.csv")

INTERNET_ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def remove_if_exists(path):
    """Delete a previous output so reruns start clean."""
    if os.path.exists(path):
        os.remove(path)
        print(f"[save] removed existing {path}")


def load_raw(path):
    df = pd.read_csv(path)
    print(f"[load] {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_identifier(df):
    """customerID identifies rows, it does not describe customers."""
    df = df.drop(columns=["customerID"])
    print("[drop] customerID removed (identifier, no predictive value)")
    return df


def fix_total_charges(df):
    """Convert TotalCharges from text to numeric.

    EDA finding: 11 rows hold a blank space instead of a number, all of them
    tenure 0 customers who have not been billed yet. The blank is a meaningful
    null (nothing billed so far), so the correct fill is 0, not an imputed
    median. Rows are kept: brand new customers are exactly what the model
    will meet in production.
    """
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blanks = int(df["TotalCharges"].isna().sum())
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"[fix] TotalCharges converted to numeric, {blanks} blanks filled with 0")
    return df


def collapse_no_service(df):
    """Collapse 'No internet service' and 'No phone service' to 'No'.

    EDA finding: these values duplicate facts already stored in
    InternetService and PhoneService. After collapsing, the fact lives once,
    in the right column, and the add-on columns become clean Yes/No.
    """
    for col in INTERNET_ADDON_COLUMNS:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    n = len(INTERNET_ADDON_COLUMNS) + 1
    print(f"[collapse] 'No X service' -> 'No' in {n} columns")
    return df


def encode_target(df):
    """Encode Churn to 1 (left) / 0 (stayed). Skipped when the column is
    absent, which is the case for prediction input (the thing being
    predicted)."""
    if "Churn" not in df.columns:
        return df
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    print(f"[target] Churn encoded to 0/1, churn rate {df['Churn'].mean():.1%}")
    return df


def check(df):
    """Sanity checks before saving.

    Duplicate rows are reported but kept: with customerID dropped, two
    different customers can legitimately share an identical setup, so
    identical rows are not errors.
    """
    nulls = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())
    assert nulls == 0, f"expected no nulls after cleaning, found {nulls}"
    if "Churn" in df.columns:
        assert set(df["Churn"].unique()) == {0, 1}, "Churn should be 0/1"
    print(f"[check] nulls: {nulls}, duplicate feature rows (kept): {dupes}")


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    remove_if_exists(path)
    df.to_csv(path, index=False)
    print(f"[save] {path}: {df.shape[0]} rows, {df.shape[1]} columns")


def transform(df):
    """Pure in-memory cleaning: the exact transforms training used, with no
    file I/O. Shared by run() (training) and predict.py (serving), so the two
    paths cannot drift apart. This is the single cleaning code path."""
    df = drop_identifier(df)
    df = fix_total_charges(df)
    df = collapse_no_service(df)
    df = encode_target(df)
    check(df)
    return df


def run(input_path=RAW_PATH, output_path=CLEANED_PATH):
    df = load_raw(input_path)
    df = transform(df)
    save(df, output_path)
    return df


if __name__ == "__main__":
    run()