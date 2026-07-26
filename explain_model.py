"""SHAP interpretability for the Telco churn project.

Explains WHY the model flags a customer as a churn risk, which is the directly
actionable part for a retention team.

Why this retrains XGBoost instead of loading the saved model: the best model
by PR-AUC was TabPFN, but the four models finished within one standard
deviation of each other, a statistical tie. TabPFN has no clean feature
attribution and cannot be explained with an exact TreeExplainer, so the
explanations are built on XGBoost instead. Since it is statistically as good
as the saved model, nothing is lost in performance and the whole
interpretability story is gained. The XGBoost is trained on the identical
split (same RANDOM_STATE) so its behaviour matches what training measured.

Global views (beeswarm, bar) show what drives churn across all customers.
Local views (waterfall) show why one specific customer was flagged.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = os.path.join("data", "processed", "telco_model_ready.csv")
ANALYSIS_DIR = "analysis"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Best XGBoost params from the training run's grid search, hardcoded so this
# script is self contained and does not re-tune. If the training grid changes,
# update these to match the reported best params.
XGB_PARAMS = dict(
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_estimators=400,
    learning_rate=0.05,
    n_jobs=1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
)


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"[save] removed existing {path}")


def load_split(path):
    """Reproduce the exact training split so the explained model sees the same
    data the training script measured."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[load] train {len(X_train)}, test {len(X_test)}, {X.shape[1]} features")
    return X_train, X_test, y_train, y_test


def train_xgb(X_train, y_train):
    from xgboost import XGBClassifier

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    print("[train] XGBoost fitted for explanation")
    return model


def explain(model, X_test):
    """TreeExplainer gives exact SHAP values for tree models, fast. The values
    are in log-odds space (the model's margin), so positive pushes toward
    churn, negative toward staying."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    print(f"[shap] computed SHAP values for {X_test.shape[0]} test customers")
    return explainer, shap_values


def save_current(filename):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    path = os.path.join(ANALYSIS_DIR, filename)
    remove_if_exists(path)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[chart] {path}")


def global_views(shap_values):
    """What drives churn across all customers. The beeswarm shows direction and
    spread per feature; the bar shows average magnitude (overall importance)."""
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("What drives churn (SHAP, all test customers)")
    save_current("shap_beeswarm.png")

    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Feature importance (mean absolute SHAP)")
    save_current("shap_bar.png")


def pick_customers(model, X_test, y_test):
    """Three instructive local cases for the retention team:
    a confident correct churn flag, a confident correct stayer, and a false
    positive (flagged but would not have churned), so the team sees what a
    wasted-offer case looks like."""
    probs = model.predict_proba(X_test)[:, 1]
    s = pd.DataFrame({"prob": probs, "actual": y_test.values}, index=range(len(y_test)))

    confident_churn = s[s["actual"] == 1].sort_values("prob", ascending=False).index[0]
    confident_stay = s[s["actual"] == 0].sort_values("prob").index[0]
    # False positive: actually stayed, but highest predicted risk among stayers
    false_positive = s[s["actual"] == 0].sort_values("prob", ascending=False).index[0]

    picks = {
        "confident_churn": confident_churn,
        "confident_stay": confident_stay,
        "false_positive": false_positive,
    }
    for label, i in picks.items():
        print(f"[pick] {label}: row {i}, "
              f"predicted {s.loc[i, 'prob']:.3f}, actual {int(s.loc[i, 'actual'])}")
    return picks


def local_views(shap_values, picks):
    """Waterfall per chosen customer: how each feature pushed this specific
    prediction up toward churn or down toward staying, starting from the
    base rate."""
    for label, i in picks.items():
        shap.plots.waterfall(shap_values[i], max_display=12, show=False)
        plt.title(f"Why this customer was scored: {label.replace('_', ' ')}")
        save_current(f"shap_waterfall_{label}.png")


def run(input_path=INPUT_PATH):
    X_train, X_test, y_train, y_test = load_split(input_path)
    model = train_xgb(X_train, y_train)
    print(model.get_booster().get_dump()[0])   # first tree as text: the split rules
    explainer, shap_values = explain(model, X_test)
    global_views(shap_values)
    picks = pick_customers(model, X_test, y_test)
    local_views(shap_values, picks)
    print("[done] SHAP charts written to analysis/")
    return model, shap_values


# shap is imported inside functions so the module imports even when shap is
# absent, but global_views/local_views need it at module scope for the plots.
import shap  # noqa: E402


if __name__ == "__main__":
    run()
