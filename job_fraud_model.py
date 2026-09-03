"""
Real / Fake Job Posting Prediction  --  binary text classification.

Dataset: Kaggle "Real or Fake: Fake Job Posting Prediction"
         https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Flow: load & clean -> feature engineering -> stratified 80/20 split ->
      GridSearchCV + 5-fold CV for LogisticRegression and RandomForest ->
      retrain best combo -> predict on the held-out test set ->
      report precision / recall / F1 on the fraudulent class.

The dataset is heavily imbalanced (~4.8% fraudulent), so accuracy is
meaningless here: every metric below is reported for the positive
(fraudulent) class, and the grid search is scored on that class's F1.

Run:
    python job_fraud_model.py            # full grids
    python job_fraud_model.py --fast     # small grids, for iterating
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "dataset" / "fake_job_postings.csv"
RESULTS_DIR = Path(__file__).parent / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
TARGET = "fraudulent"

# Text fields merged into one document per posting.
TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]

# Fields whose absence is itself a signal: scam postings very often leave the
# company profile and requirements empty.
MISSING_FLAG_SOURCES = [
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "department",
    "salary_range",
]

# Already-binary columns that ship with the dataset.
BINARY_COLS = ["telecommuting", "has_company_logo", "has_questions"]

# Low-cardinality columns for one-hot encoding. `country` is derived from
# `location`, which is far too high-cardinality to encode raw.
CAT_COLS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "country",
]

META_COLS = (
    ["text_len", "word_count", "description_len", "n_missing_fields"]
    + [f"{c}_missing" for c in MISSING_FLAG_SOURCES]
    + BINARY_COLS
)

FEATURE_COLS = ["combined_text"] + META_COLS + CAT_COLS


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# 1. Load & clean
# --------------------------------------------------------------------------

def load_and_clean(path: Path = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    """Load the raw CSV and return a tidy frame of features + target.

    Cleaning steps:
      * drop `job_id` (a pure row identifier -- no signal, and a leakage risk)
      * record which fields were missing *before* filling them
      * fill missing text with "" and missing categories with "Unknown"
      * collapse `location` down to a country code
      * build the combined text document and the numeric meta features
    """
    df = pd.read_csv(path)

    if verbose:
        banner("1. LOAD & CLEAN")
        print(f"Raw shape: {df.shape}")
        print("\nMissing values per column:")
        print(df.isna().sum().sort_values(ascending=False).to_string())

    # --- drop the identifier -------------------------------------------------
    df = df.drop(columns=["job_id"], errors="ignore")

    # --- missingness flags, captured BEFORE we fill anything -----------------
    out = pd.DataFrame(index=df.index)
    for col in MISSING_FLAG_SOURCES:
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        out[f"{col}_missing"] = blank.astype(int)
    out["n_missing_fields"] = out[[f"{c}_missing" for c in MISSING_FLAG_SOURCES]].sum(axis=1)

    # --- text fields: fill and normalise whitespace --------------------------
    text = {}
    for col in TEXT_FIELDS:
        text[col] = (
            df[col].fillna("").astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    out["combined_text"] = (
        text["title"] + " " + text["company_profile"] + " " + text["description"]
        + " " + text["requirements"] + " " + text["benefits"]
    ).str.strip()

    # --- numeric meta features ----------------------------------------------
    out["text_len"] = out["combined_text"].str.len()
    out["word_count"] = out["combined_text"].str.split().str.len()
    out["description_len"] = text["description"].str.len()

    # --- categorical fields --------------------------------------------------
    # "US, NY, New York" -> "US"
    out["country"] = (
        df["location"].fillna("").astype(str)
        .str.split(",").str[0].str.strip().str.upper()
        .replace("", "Unknown")
    )
    for col in CAT_COLS:
        if col == "country":
            continue
        out[col] = (
            df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        )

    # --- binaries and target -------------------------------------------------
    for col in BINARY_COLS:
        out[col] = df[col].fillna(0).astype(int)

    out[TARGET] = df[TARGET].astype(int)
    assert out[TARGET].notna().all(), "target contains NaNs"

    if verbose:
        n_pos = int(out[TARGET].sum())
        print(f"\nCleaned shape: {out.shape}  (job_id dropped)")
        print(
            f"Class balance: {len(out) - n_pos} real / {n_pos} fake "
            f"({n_pos / len(out):.2%} fraudulent)  ->  imbalanced"
        )
        print("\nFirst 5 cleaned rows:")
        print(out[["text_len", "word_count", "n_missing_fields", "country", TARGET]]
              .head().to_string())

    return out


# --------------------------------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------------------------------

def build_feature_transformer(max_features: int = 50_000, min_df: int = 3,
                              ngram_range: tuple = (1, 2)) -> ColumnTransformer:
    """Assemble the three feature blocks into one ColumnTransformer.

    Keeping the vectorizer / scaler / encoder *inside* the transformer (which
    in turn lives inside a Pipeline) is what makes the cross-validation honest:
    each of them is fit only on the training folds, never on validation data.

      text  -> TF-IDF over combined title+profile+description+reqs+benefits
      meta  -> standard-scaled lengths, missing-field flags and binaries
      cat   -> one-hot encoded low-cardinality columns
    """
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    sublinear_tf=True,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    stop_words="english",
                    strip_accents="unicode",
                ),
                "combined_text",  # a single column -> the vectorizer gets a 1-D series
            ),
            ("meta", StandardScaler(), META_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def show_feature_engineering(fitted_pipeline: Pipeline, X: pd.DataFrame,
                             name: str, n: int = 3) -> None:
    """Trace n example rows through feature engineering (assignment part 2)."""
    banner(f"FEATURE ENGINEERING TRACE -- {n} examples from {name}")
    ct = fitted_pipeline.named_steps["features"]
    Z = ct.transform(X.head(n))
    vec = ct.named_transformers_["text"]
    vocab = np.array(vec.get_feature_names_out())

    for i in range(n):
        row = X.iloc[i]
        print(f"\n--- example {i} ---")
        print(f"  raw combined_text[:160]: {row['combined_text'][:160]!r}")
        print(f"  meta: text_len={row['text_len']}, word_count={row['word_count']}, "
              f"n_missing_fields={row['n_missing_fields']}")
        print(f"  cat : employment_type={row['employment_type']!r}, "
              f"country={row['country']!r}")
        dense = Z[i].toarray().ravel() if hasattr(Z, "toarray") else np.asarray(Z[i]).ravel()
        text_part = dense[: len(vocab)]
        top = np.argsort(text_part)[::-1][:8]
        n_onehot = dense.shape[0] - len(vocab) - len(META_COLS)
        print(f"  -> vector length {dense.shape[0]} "
              f"({len(vocab)} tf-idf + {len(META_COLS)} meta + {n_onehot} one-hot)")
        print("  -> top tf-idf terms: "
              + ", ".join(f"{vocab[j]}={text_part[j]:.3f}" for j in top if text_part[j] > 0))


# --------------------------------------------------------------------------
# 3. Models + grids
# --------------------------------------------------------------------------

def make_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("features", build_feature_transformer(max_features=50_000, min_df=3)),
        ("clf", LogisticRegression(
            solver="liblinear",   # strong on sparse, high-dimensional text
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])


def make_rf_pipeline() -> Pipeline:
    # A forest cannot exploit 50k sparse columns the way a linear model can,
    # and it costs a lot of time to try -- so give it a lighter vocabulary.
    return Pipeline([
        ("features", build_feature_transformer(max_features=20_000, min_df=5,
                                               ngram_range=(1, 1))),
        ("clf", RandomForestClassifier(
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def get_grids(fast: bool) -> dict:
    """Hyperparameter grids. The full cartesian product of each is searched."""
    if fast:
        return {
            "LogisticRegression": {"clf__C": [0.1, 1],
                                   "clf__class_weight": [None, "balanced"]},
            "RandomForest": {"clf__n_estimators": [100],
                             "clf__max_depth": [None, 20]},
        }
    return {
        "LogisticRegression": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced"],
        },
        "RandomForest": {
            "clf__n_estimators": [100, 300, 500],
            "clf__max_depth": [None, 20, 50],
        },
    }


# --------------------------------------------------------------------------
# 4. Train (grid search + 5-fold CV)
# --------------------------------------------------------------------------

def train(pipeline: Pipeline, grid: dict, X_train: pd.DataFrame, y_train: pd.Series,
          name: str, n_jobs: int = -1):
    """Grid-search `pipeline` over `grid` with stratified 5-fold CV.

    Scored on F1 of the positive (fraudulent) class -- the metric the
    assignment specifies for a binary problem with a single class of interest.
    GridSearchCV refits the winning combination on the full training set.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, grid, scoring="f1", cv=cv, n_jobs=n_jobs, refit=True,
        verbose=1, return_train_score=False,
    )

    banner(f"4. GRID SEARCH + {N_FOLDS}-FOLD CV -- {name}")
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    print(f"Grid: {grid}")
    print(f"{n_combos} combinations x {N_FOLDS} folds = {n_combos * N_FOLDS} fits")

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    param_cols = [c for c in search.cv_results_ if c.startswith("param_")]
    results = (
        pd.DataFrame(search.cv_results_)[
            param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
        ]
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )
    print(f"\nAll permutations, mean CV F1 over {N_FOLDS} folds ({elapsed:.1f}s):")
    print(results.to_string(index=False))
    print(f"\nBEST: {search.best_params_}   mean CV F1 = {search.best_score_:.4f}")
    return search, results


def predict(model, X: pd.DataFrame):
    """Return (hard labels, probability of the fraudulent class)."""
    return model.predict(X), model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# 5. Evaluate on the held-out test set
# --------------------------------------------------------------------------

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    y_pred, y_proba = predict(model, X_test)

    metrics = {
        "model": name,
        "precision_fraudulent": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_fraudulent": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_fraudulent": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }

    banner(f"5. TEST-SET EVALUATION -- {name}")
    print("Metrics for the FRAUDULENT (positive) class:")
    print(f"  precision = {metrics['precision_fraudulent']:.4f}")
    print(f"  recall    = {metrics['recall_fraudulent']:.4f}")
    print(f"  F1        = {metrics['f1_fraudulent']:.4f}")
    print(f"  (support: ROC-AUC = {metrics['roc_auc']:.4f}, "
          f"PR-AUC = {metrics['pr_auc']:.4f})")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["real (0)", "fake (1)"],
                                digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=["actual real", "actual fake"],
                       columns=["pred real", "pred fake"]).to_string())
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def sample_predictions(model, X_test: pd.DataFrame, y_test: pd.Series, name: str,
                       n_first: int = 5) -> pd.DataFrame:
    """First n test predictions (required), plus a look at hits and misses."""
    y_pred, y_proba = predict(model, X_test)
    table = pd.DataFrame({
        "text_snippet": X_test["combined_text"].str.slice(0, 55).values,
        "actual": y_test.values,
        "predicted": y_pred,
        "P(fake)": np.round(y_proba, 4),
        "correct": y_test.values == y_pred,
    }, index=X_test.index)

    banner(f"SAMPLE PREDICTIONS -- {name}")
    print(f"First {n_first} rows of the test set:")
    print(table.head(n_first).to_string())

    groups = [
        ("True positives (caught fakes)", (table.actual == 1) & (table.predicted == 1)),
        ("False positives (real flagged as fake)", (table.actual == 0) & (table.predicted == 1)),
        ("False negatives (fakes that slipped through)", (table.actual == 1) & (table.predicted == 0)),
    ]
    for label, mask in groups:
        part = table[mask].head(3)
        print(f"\n{label}:")
        print(part.to_string() if len(part) else "  (none)")

    return table


def plot_confusion_matrices(metrics_list: list, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics_list), figsize=(5 * len(metrics_list), 4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics_list):
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{m['model']}\nF1(fake) = {m['f1_fraudulent']:.3f}")
        ax.set_xticks([0, 1], ["pred real", "pred fake"])
        ax.set_yticks([0, 1], ["actual real", "actual fake"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def split_data(df: pd.DataFrame, verbose: bool = True):
    """Stratified 80/20 split -- stratify is essential at a ~4.8% positive rate."""
    X, y = df[FEATURE_COLS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    if verbose:
        banner("3. STRATIFIED 80/20 SPLIT")
        print(f"train: {X_train.shape[0]:,} rows, {y_train.mean():.2%} fraudulent")
        print(f"test : {X_test.shape[0]:,} rows, {y_test.mean():.2%} fraudulent")
        print("(stratify=y keeps the positive rate identical in both halves)")
        cols = ["text_len", "n_missing_fields", "employment_type", "country"]
        print("\nFirst 5 TRAIN rows:")
        print(X_train[cols].head().to_string())
        print("\nFirst 5 TEST rows:")
        print(X_test[cols].head().to_string())
    return X_train, X_test, y_train, y_test


def run(fast: bool = False, data_path: Path = DATA_PATH,
        results_dir: Path = RESULTS_DIR) -> dict:
    results_dir.mkdir(exist_ok=True)

    df = load_and_clean(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    grids = get_grids(fast)
    specs = [
        ("LogisticRegression", make_logreg_pipeline(), -1),
        # RF already parallelises over trees; nesting n_jobs=-1 would oversubscribe.
        ("RandomForest", make_rf_pipeline(), 1),
    ]

    all_metrics, all_cv, traced = [], {}, False
    for name, pipe, n_jobs in specs:
        search, cv_table = train(pipe, grids[name], X_train, y_train, name, n_jobs=n_jobs)
        all_cv[name] = cv_table
        cv_table.to_csv(results_dir / f"cv_results_{name}.csv", index=False)

        if not traced:  # show the FE trace once, on the fitted best estimator
            show_feature_engineering(search.best_estimator_, X_train, "TRAIN")
            show_feature_engineering(search.best_estimator_, X_test, "TEST")
            traced = True

        m = evaluate(search.best_estimator_, X_test, y_test, name)
        m["best_params"] = {k: str(v) for k, v in search.best_params_.items()}
        m["best_cv_f1"] = float(search.best_score_)
        all_metrics.append(m)

        table = sample_predictions(search.best_estimator_, X_test, y_test, name)
        table.to_csv(results_dir / f"sample_predictions_{name}.csv")
        joblib.dump(search.best_estimator_, results_dir / f"model_{name}.joblib")

    # --- comparison ----------------------------------------------------------
    comparison = pd.DataFrame([{
        "model": m["model"],
        "best_params": m["best_params"],
        "cv_f1_mean": round(m["best_cv_f1"], 4),
        "test_precision": round(m["precision_fraudulent"], 4),
        "test_recall": round(m["recall_fraudulent"], 4),
        "test_f1": round(m["f1_fraudulent"], 4),
        "test_roc_auc": round(m["roc_auc"], 4),
        "test_pr_auc": round(m["pr_auc"], 4),
    } for m in all_metrics])

    banner("MODEL COMPARISON (fraudulent class)")
    print(comparison.to_string(index=False))
    winner = comparison.loc[comparison.test_f1.idxmax(), "model"]
    print(f"\nWinner on test F1: {winner}")

    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    (results_dir / "metrics.json").write_text(
        json.dumps({"models": all_metrics, "winner": winner}, indent=2), encoding="utf-8")
    plot_confusion_matrices(all_metrics, results_dir / "confusion_matrices.png")
    print(f"\nArtifacts written to {results_dir}")

    return {"comparison": comparison, "metrics": all_metrics, "cv": all_cv, "winner": winner}


def main() -> None:
    ap = argparse.ArgumentParser(description="Real/fake job posting classifier")
    ap.add_argument("--fast", action="store_true", help="small grids, for quick iteration")
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    args = ap.parse_args()
    run(fast=args.fast, data_path=args.data)


if __name__ == "__main__":
    main()
