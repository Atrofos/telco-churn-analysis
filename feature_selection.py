"""Feature selection and encoding for the Telco churn project.

Takes the cleaned dataset and produces a fully numeric, model-ready dataset.
Encoding choices follow the EDA: ordered categories get ordinal maps that
preserve their order, unordered categories get one-hot so no false ordering
is implied, and Yes/No columns become 0/1.

Selection follows the variance / impact / redundancy framework: check for
near-constant columns, look at each feature's relationship with the target,
and drop the weaker of highly correlated pairs.
"""

import os

import pandas as pd

INPUT_PATH = os.path.join("data", "processed", "telco_cleaned.csv")
OUTPUT_PATH = os.path.join("data", "processed", "telco_model_ready.csv")

# Yes/No columns that become 1/0.
BINARY_YES_NO = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

# One-hot columns, so no false ordering is implied.
#
# Contract is ordered (Month-to-month, One year, Two year), so ordinal was the
# obvious choice. It is one-hot instead because ordinal encoding asserts equal
# spacing between levels, and the EDA shows the spacing is nowhere near equal:
# churn runs 42.7% -> 11.3% -> 2.8%, a collapsing curve, not even steps. Trees
# ignore the difference (they only split on order), but the logistic baseline
# reads the spacing literally, so one-hot keeps its coefficients honest. With
# only three levels the order information lost to one-hot is negligible.
ONE_HOT_COLUMNS = ["InternetService", "PaymentMethod", "Contract"]

# Redundancy decision: TotalCharges is tenure x MonthlyCharges plus symmetric
# noise. The EDA tested whether the residual (actual total minus the naive
# product) carried any churn signal, in case bill changes over a customer's
# life mattered. It did not: the residual correlated 0.000 with churn and sat
# symmetric around zero, the signature of rounding and part-month noise rather
# than a real effect. So the column is genuinely redundant, not just
# correlated, and dropping it removes multicollinearity at no information cost.
# Listed as a constant so the decision is visible and easy to reverse.
DROP_REDUNDANT = ["TotalCharges"]


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"[save] removed existing {path}")


def load_cleaned(path):
    df = pd.read_csv(path)
    print(f"[load] {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def encode_binary(df):
    df["gender"] = (df["gender"] == "Male").astype(int)
    for col in BINARY_YES_NO:
        df[col] = (df[col] == "Yes").astype(int)
    print(f"[encode] gender and {len(BINARY_YES_NO)} Yes/No columns -> 0/1")
    return df


def encode_one_hot(df):
    before = df.shape[1]
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, dtype=int)
    print(f"[encode] one-hot {ONE_HOT_COLUMNS}: {before} -> {df.shape[1]} columns")
    return df


def variance_check(df):
    """Report near-constant columns (a feature that never varies cannot
    predict anything). Nothing in this dataset is expected to fail."""
    features = df.drop(columns=["Churn"]) if "Churn" in df.columns else df
    dominant = (features.apply(lambda c: c.value_counts(normalize=True).iloc[0])
                .sort_values(ascending=False))
    near_constant = dominant[dominant > 0.99]
    if near_constant.empty:
        print("[check] variance: no near-constant columns")
    else:
        print(f"[check] variance: near-constant columns found:\n{near_constant}")
    return df


def impact_check(df):
    """Informational: correlation of each feature with the target. Point
    biserial correlation is crude (it only sees linear relationships) but a
    useful first sanity check that the EDA findings survived encoding.
    Target dependent, so skipped when Churn is absent (prediction input)."""
    if "Churn" not in df.columns:
        return df
    corr = (df.corr(numeric_only=True)["Churn"]
            .drop("Churn")
            .sort_values(key=abs, ascending=False))
    print("[check] impact: top correlations with Churn")
    print(corr.head(8).round(3).to_string())
    return df


def redundancy_check(df):
    """Show the evidence behind DROP_REDUNDANT, then apply it. The correlation
    print is target independent, but only runs when the columns are present."""
    pairs = [("tenure", "TotalCharges"), ("MonthlyCharges", "TotalCharges")]
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            print(f"[check] redundancy: corr({a}, {b}) = {df[a].corr(df[b]):.3f}")
    df = df.drop(columns=[c for c in DROP_REDUNDANT if c in df.columns])
    print(f"[drop] redundant: {DROP_REDUNDANT}")
    return df


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    remove_if_exists(path)
    df.to_csv(path, index=False)
    print(f"[save] {path}: {df.shape[0]} rows, {df.shape[1]} columns")


def transform(df):
    """Pure in-memory encoding and selection: the exact transforms training
    used, no file I/O. Shared by run() (training) and predict.py (serving) so
    the encoding cannot drift between the two. The target-dependent checks
    (impact) no-op when Churn is absent, so this runs on prediction input too.

    Column alignment (making sure prediction data ends up with exactly the
    training columns in the training order) is not done here. It is the
    predictor's job, using the feature order saved in the model bundle, since
    a single batch may be missing one-hot categories that training saw."""
    df = encode_binary(df)
    df = encode_one_hot(df)
    df = variance_check(df)
    df = impact_check(df)
    df = redundancy_check(df)
    return df


def run(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    df = load_cleaned(input_path)
    df = transform(df)
    save(df, output_path)
    return df


if __name__ == "__main__":
    run()