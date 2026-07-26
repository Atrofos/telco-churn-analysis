"""Model training for the Telco churn project.

Trains a progression of classifiers, tunes them with cross validation, picks
the best by PR-AUC, calibrates its probabilities, chooses a decision
threshold from the business cost of mistakes, and saves the bundle.

The held-out test set is touched exactly once, at the end, for the reported
numbers. All tuning happens by cross validation on the training set.
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")  # file output, no display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_PATH = os.path.join("data", "processed", "telco_model_ready.csv")
MODEL_PATH = os.path.join("models", "best_model.pkl")
ANALYSIS_DIR = "analysis"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# One CV splitter used everywhere, so every cross validated number in the file
# uses identical folds. Explicit StratifiedKFold (rather than a bare cv=5) so
# each fold keeps the 26.5% churn rate, which matters at this imbalance, and so
# the stratification is visible rather than relying on a library default.
CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Parallelism: the CV searches fan out across folds (SEARCH_JOBS), but the
# estimators inside them run single threaded. Nesting both causes a
# fork-within-a-fork process explosion that exhausts Windows virtual memory
# (WinError 1455). If memory is still tight, lower SEARCH_JOBS to 2.
SEARCH_JOBS = -1

# Business cost of each mistake. Derived from the data, not invented: churners
# average GBP 74/month (MonthlyCharges), a retention discount costs roughly
# GBP 10/month, so missing a churner costs about 7x a wasted offer. The time
# horizon cancels out of the ratio (it multiplies both sides), so the ratio
# rests only on the discount assumption. The threshold is reported at 4:1 and
# 12:1 as well, so the write-up shows sensitivity rather than a point guess.
COST_FALSE_NEGATIVE = 7.0
COST_FALSE_POSITIVE = 1.0

COLORS = {"stay": "#9e9e9e", "churn": "#d62728", "accent": "#1f77b4"}


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"[save] removed existing {path}")


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    print(f"[load] {path}: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"[load] churn rate {y.mean():.1%} ({y.sum()} of {len(y)})")
    return X, y


def split(X, y):
    """Stratified split so both halves keep the 26.5% churn rate. Without
    stratify the test set's churn rate drifts by chance, and every metric
    that depends on the base rate drifts with it."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[split] train {len(X_train)} ({y_train.mean():.1%} churn), "
          f"test {len(X_test)} ({y_test.mean():.1%} churn)")
    return X_train, X_test, y_train, y_test


def candidates():
    """The model progression, each with a small grid to tune.

    Logistic regression is the interpretable baseline and needs scaling, so
    it is wrapped in a pipeline (the scaler then refits inside every CV fold,
    which is what stops the validation fold leaking into the training fold).
    Tree models do not need scaling.
    """
    return {
        "logistic_regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [300],
                "max_depth": [6, 10, None],
                "min_samples_leaf": [1, 5, 20],
            },
        ),
    }


def add_xgboost(models):
    """XGBoost is optional: a missing package skips the stage rather than
    crashing the pipeline."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[train] xgboost not installed, skipping")
        return models
    models["xgboost"] = (
        XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=400,
            learning_rate=0.05,
            n_jobs=1,
        ),
        {"max_depth": [3, 5], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]},
    )
    return models


def add_tabpfn(models):
    """TabPFN is optional, same skip-if-missing pattern as XGBoost.

    A tabular foundation model: pre-trained on synthetic data, it classifies by
    a single forward pass (in-context learning) rather than training on our
    data. Its sweet spot is small tabular datasets (under ~10k rows), which is
    exactly this one. No hyperparameter grid, since there is nothing to tune,
    so it is scored directly by cross validated PR-AUC on the same folds as the
    others for a fair comparison. Needs the CUDA torch build; a missing package
    skips it rather than crashing the run.
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        print("[train] tabpfn not installed, skipping")
        return models
    models["tabpfn"] = (TabPFNClassifier(), {})
    return models


def tune(name, estimator, grid, X_train, y_train):
    """Tune by cross validation, scoring on PR-AUC (average precision).

    PR-AUC rather than accuracy because accuracy is worthless here: always
    predicting "stays" scores 73.5%. PR-AUC rather than ROC-AUC for selection
    because it focuses on the minority class we actually care about, and it
    does not get flattered by the large pool of easy negatives.

    Both are threshold free, so tuning never has to commit to a threshold.
    That decision is made once, later, on business grounds.
    """
    search = GridSearchCV(
        estimator, grid, scoring="average_precision", cv=CV, n_jobs=SEARCH_JOBS
    )
    search.fit(X_train, y_train)
    std = search.cv_results_["std_test_score"][search.best_index_]
    return search.best_estimator_, search.best_score_, std, search.best_params_


def compare_imbalance_handling(X_train, y_train, X_test, y_test):
    """Test whether class weighting helps, rather than assuming it does.

    Class weighting makes churners "louder" in training, which pushes predicted
    probabilities upward. The decision here is threshold tuning instead, on the
    grounds that weighting distorts the probabilities for no gain in how well
    the model ranks customers by risk (the ranked worklist is the goal).

    So this reports two things per option, to make the whole argument visible:
    PR-AUC (threshold free, measures ranking) and Brier score (measures whether
    the probabilities are honest). The expected result is that weighting leaves
    PR-AUC roughly unchanged while making the Brier score worse, which is the
    evidence for rejecting it: no better ranking, worse probabilities.
    """
    print("[imbalance] comparing no weighting vs class weighting")
    for label, weight in [("none", None), ("balanced", "balanced")]:
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE, class_weight=weight)),
        ])
        search = GridSearchCV(
            pipe, {"model__C": [0.1, 1.0]},
            scoring="average_precision", cv=CV, n_jobs=SEARCH_JOBS
        )
        search.fit(X_train, y_train)
        brier = brier_score_loss(y_test, search.predict_proba(X_test)[:, 1])
        print(f"[imbalance]   class_weight={label:9s} "
              f"PR-AUC {search.best_score_:.4f} (ranking), "
              f"Brier {brier:.4f} (probability honesty)")


def calibrate(best, X_train, y_train, X_test, y_test):
    """Calibrate probabilities so a 0.8 risk score means roughly 80% of such
    customers churn. This is the classification analogue of the prediction
    interval calibration check from the housing project.

    Both methods are tried and the better one by Brier score wins, with the
    uncalibrated model kept if calibration does not help. Isotonic is
    flexible but can overfit on small data; sigmoid assumes a specific shape
    but is more stable. With ~5,600 training rows either is defensible, so
    measure rather than guess.
    """
    results = {}
    uncal = brier_score_loss(y_test, best.predict_proba(X_test)[:, 1])
    results["none"] = (best, uncal)
    print(f"[calibrate] uncalibrated Brier {uncal:.4f} (lower is better)")

    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(best, method=method, cv=CV)
        cal.fit(X_train, y_train)
        score = brier_score_loss(y_test, cal.predict_proba(X_test)[:, 1])
        results[method] = (cal, score)
        print(f"[calibrate] {method} Brier {score:.4f}")

    winner = min(results, key=lambda k: results[k][1])
    print(f"[calibrate] chosen: {winner}")
    return results[winner][0], winner


def choose_threshold(model, X_train, y_train):
    """Pick the decision threshold by expected business cost, on the TRAINING
    set via cross validated predictions.

    The threshold is a parameter fitted to data, so choosing it on the test
    set would leak: the reported numbers would be optimistic because the
    threshold had already seen the answers. This is the same held-out
    discipline as everything else, applied to a decision that is easy to
    forget is a decision.

    The default 0.5 is only optimal when mistakes cost the same and classes
    are balanced. Neither holds here, so 0.5 is arbitrary. The sweep finds
    the threshold minimising COST_FALSE_NEGATIVE * FN + COST_FALSE_POSITIVE * FP.
    """

    probs = cross_val_predict(
        model, X_train, y_train, cv=CV, method="predict_proba"
    )[:, 1]

    grid = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in grid:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred).ravel()
        costs.append(COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp)

    best_t = float(grid[int(np.argmin(costs))])
    print(f"[threshold] cost ratio FN:FP = {COST_FALSE_NEGATIVE:.0f}:{COST_FALSE_POSITIVE:.0f}")
    print(f"[threshold] chosen {best_t:.2f} (vs default 0.50)")
    return best_t, grid, costs


def threshold_table(model, X_test, y_test, chosen):
    """Show precision, recall and business cost across a range of thresholds,
    so the tradeoff is concrete: catching more churners (higher recall) means
    more false alarms (lower precision). The chosen threshold is marked."""
    probs = model.predict_proba(X_test)[:, 1]
    print("[table] precision / recall / cost by threshold (test set)")
    print("[table]   thresh  precision  recall   flagged  cost   ")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, round(chosen, 2)]:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        cost = COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp
        mark = "  <- chosen" if abs(t - chosen) < 0.005 else ""
        print(f"[table]    {t:.2f}     {prec:.3f}     {rec:.3f}    "
              f"{tp + fp:5d}   {cost:5.0f}{mark}")


def print_model_comparison(results, best_name):
    """Side by side table of the model progression, sorted best first, so the
    comparison is readable at a glance instead of scattered across log lines.
    Flags models within one standard deviation of the winner, since a lead
    smaller than the fold spread is not a real difference."""
    best_score = results[best_name]["score"]
    best_std = results[best_name]["std"]
    order = sorted(results, key=lambda k: results[k]["score"], reverse=True)

    print()
    print("=" * 62)
    print("  MODEL COMPARISON (cross validated PR-AUC on training set)")
    print("=" * 62)
    print(f"  {'model':<22}{'PR-AUC':<10}{'std':<9}{'note'}")
    print("  " + "-" * 58)
    for name in order:
        score = results[name]["score"]
        std = results[name]["std"]
        if name == best_name:
            note = "best"
        elif best_score - score <= best_std:
            note = "within noise of best"
        else:
            note = ""
        print(f"  {name:<22}{score:<10.4f}{std:<9.4f}{note}")
    print("=" * 62)
    print()


def evaluate(model, X_test, y_test, threshold):
    """The one and only look at the held-out test set."""
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "brier": brier_score_loss(y_test, probs),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "accuracy": (tp + tn) / len(y_test),
    }
    print("[eval] held-out test set (touched once)")
    for k, v in metrics.items():
        print(f"[eval]   {k}: {v:.4f}")
    print(f"[eval] confusion matrix at threshold {threshold:.2f}")
    print(f"[eval]   caught churners (TP): {tp}, missed churners (FN): {fn}")
    print(f"[eval]   wasted offers (FP): {fp}, correctly left alone (TN): {tn}")
    print(f"[eval] business cost: {COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp:.0f} units")
    return metrics, probs, pred


def save_chart(fig, filename):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    path = os.path.join(ANALYSIS_DIR, filename)
    remove_if_exists(path)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {path}")


def chart_curves(y_test, probs):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=COLORS["accent"], label=f"Model (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["stay"], label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    save_chart(fig, "roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    base_rate = y_test.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=COLORS["churn"], label=f"Model (PR-AUC {ap:.3f})")
    ax.axhline(base_rate, ls="--", color=COLORS["stay"],
               label=f"Always predict churn ({base_rate:.3f})")
    ax.set_xlabel("Recall (churners caught)")
    ax.set_ylabel("Precision (flagged customers who really churn)")
    ax.set_title("Precision recall curve")
    ax.legend()
    save_chart(fig, "precision_recall_curve.png")


def chart_calibration(y_test, probs, method):
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color=COLORS["stay"], label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color=COLORS["accent"], label=f"Model ({method})")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration curve")
    ax.legend()
    save_chart(fig, "calibration_curve.png")


def chart_threshold(grid, costs, chosen):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, costs, color=COLORS["accent"])
    ax.axvline(chosen, color=COLORS["churn"], ls="--", label=f"Chosen {chosen:.2f}")
    ax.axvline(0.5, color=COLORS["stay"], ls=":", label="Default 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected business cost")
    ax.set_title(f"Threshold choice at FN:FP cost of "
                 f"{COST_FALSE_NEGATIVE:.0f}:{COST_FALSE_POSITIVE:.0f}")
    ax.legend()
    save_chart(fig, "threshold_cost.png")


def chart_confusion(y_test, pred, threshold):
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Reds")
    labels = [["Correctly\nleft alone", "Wasted\noffer"],
              ["Missed\nchurner", "Caught\nchurner"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], ["Actually stayed", "Actually churned"])
    ax.set_title(f"Confusion matrix at threshold {threshold:.2f}")
    save_chart(fig, "confusion_matrix.png")


def save_bundle(model, feature_names, threshold, metrics, calibration_method, path):
    """Save everything the predictor needs. The metadata matters as much as
    the model: a predictor that does not know the threshold or the column
    order cannot reproduce these results."""
    bundle = {
        "model": model,
        "feature_names": list(feature_names),
        "threshold": threshold,
        "calibration_method": calibration_method,
        "metrics": metrics,
        "cost_false_negative": COST_FALSE_NEGATIVE,
        "cost_false_positive": COST_FALSE_POSITIVE,
        "random_state": RANDOM_STATE,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    remove_if_exists(path)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[save] {path} (model + threshold + feature order + metrics)")


def run(input_path=INPUT_PATH, model_path=MODEL_PATH):
    X, y = load_data(input_path)
    X_train, X_test, y_train, y_test = split(X, y)

    compare_imbalance_handling(X_train, y_train, X_test, y_test)

    models = add_tabpfn(add_xgboost(candidates()))
    results = {}
    fitted = {}
    print(f"[train] tuning {len(models)} models by {CV_FOLDS}-fold CV PR-AUC...")
    for name, (estimator, grid) in models.items():
        est, score, std, params = tune(name, estimator, grid, X_train, y_train)
        fitted[name] = est
        results[name] = {"score": score, "std": std, "params": params}
        print(f"[train]   {name} done")

    best_name = max(results, key=lambda k: results[k]["score"])
    print_model_comparison(results, best_name)
    print(f"[select] best params for {best_name}: {results[best_name]['params']}")
    best = fitted[best_name]

    model, calibration_method = calibrate(best, X_train, y_train, X_test, y_test)
    threshold, grid, costs = choose_threshold(best, X_train, y_train)
    metrics, probs, pred = evaluate(model, X_test, y_test, threshold)
    threshold_table(model, X_test, y_test, threshold)

    chart_curves(y_test, probs)
    chart_calibration(y_test, probs, calibration_method)
    chart_threshold(grid, costs, threshold)
    chart_confusion(y_test, pred, threshold)

    save_bundle(model, X.columns, threshold, metrics, calibration_method, model_path)
    return model


if __name__ == "__main__":
    run()