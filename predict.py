"""Batch churn prediction for the Telco project.

Takes a raw CSV of customers (same shape as the original Telco file, with or
without a Churn column) and outputs each customer's churn probability plus a
yes/no flag at the threshold chosen during training.

Anti-skew discipline: this reuses the exact transform functions from
data_cleaning and feature_selection, so serving and training share one code
path and cannot drift. New data is then aligned to the feature order saved in
the model bundle, so a batch missing a one-hot category (say no two-year
contracts this week) still lines up with what the model expects.
"""

import os
import pickle

import pandas as pd

import data_cleaning
import feature_selection

MODEL_PATH = os.path.join("models", "best_model.pkl")
DEFAULT_INPUT = os.path.join("data", "test", "new_customers.csv")
DEFAULT_OUTPUT = os.path.join("data", "test", "predictions.csv")


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"[save] removed existing {path}")


def load_bundle(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[load] model bundle from {path}")
    print(f"[load]   threshold {bundle['threshold']:.2f}, "
          f"{len(bundle['feature_names'])} features expected")
    return bundle


def load_raw(path):
    df = pd.read_csv(path)
    print(f"[load] {path}: {df.shape[0]} customers")
    return df


def prepare(df):
    """Run the incoming raw data through the shared training transforms.

    customerID is dropped inside data_cleaning.transform, so it is captured
    first if present, to label the output. Any Churn column is ignored for
    prediction (dropped after transform), since predicting it is the point.
    """
    ids = df["customerID"].copy() if "customerID" in df.columns else None
    df = data_cleaning.transform(df)
    df = feature_selection.transform(df)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    return df, ids


def align(df, feature_names):
    """Reindex to the exact training columns in the exact training order.

    Missing columns (a one-hot category absent from this batch) are filled
    with 0. Unexpected columns (something the model never saw) are dropped.
    This is what stops a batch silently misaligning against the model.
    """
    missing = [c for c in feature_names if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_names]
    if missing:
        print(f"[align] filling {len(missing)} absent columns with 0: {missing}")
    if extra:
        print(f"[align] dropping {len(extra)} unexpected columns: {extra}")
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


def predict(bundle, X):
    model = bundle["model"]
    threshold = bundle["threshold"]
    probs = model.predict_proba(X)[:, 1]
    flags = (probs >= threshold).astype(int)
    print(f"[predict] {len(X)} customers scored, "
          f"{int(flags.sum())} flagged at threshold {threshold:.2f}")
    return probs, flags


def build_output(ids, probs, flags):
    out = pd.DataFrame({
        "churn_probability": probs.round(4),
        "churn_flag": flags,
    })
    if ids is not None:
        out.insert(0, "customerID", ids.values)
    out = out.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    return out


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    remove_if_exists(path)
    df.to_csv(path, index=False)
    print(f"[save] {path}: {df.shape[0]} rows")


def run(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, model_path=MODEL_PATH):
    bundle = load_bundle(model_path)
    raw = load_raw(input_path)
    X, ids = prepare(raw)
    X = align(X, bundle["feature_names"])
    probs, flags = predict(bundle, X)
    out = build_output(ids, probs, flags)
    save(out, output_path)
    print("[done] ranked worklist ready, highest risk first")
    return out


if __name__ == "__main__":
    run()