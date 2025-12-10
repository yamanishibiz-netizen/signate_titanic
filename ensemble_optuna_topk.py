r"""Ensemble top-K Optuna trials: evaluate by CV and produce averaged submission.

Usage (PowerShell):
    & 'C:/Users/enosh/anaconda3/python.exe' 'ensemble_optuna_topk.py' --top_k 5

This script tries to read `optuna_trials.csv` in the same folder (as produced
by `study.trials_dataframe().to_csv(...)`). If not found, it falls back to a
small internal param grid.
"""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import argparse


def load_data(script_dir: Path):
    def load_candidate(stems):
        for s in stems:
            for ext in (".csv", ".parquet", ".pkl"):
                p = script_dir / (s + ext)
                if p.exists():
                    if ext == ".csv":
                        return pd.read_csv(p)
                    elif ext == ".parquet":
                        return pd.read_parquet(p)
                    else:
                        return pd.read_pickle(p)
        return None

    train = load_candidate(["train", "train_data", "train_df", "train_final"]) 
    test = load_candidate(["test", "test_data", "test_df", "test_final"]) 
    if train is None or test is None:
        raise RuntimeError("train/test not found in script dir")
    return train, test


def ensure_features(df: pd.DataFrame):
    df = df.copy()
    if "embarked" in df.columns:
        d = pd.get_dummies(df["embarked"], prefix="embarked")
        for col in ("embarked_S", "embarked_C", "embarked_Q"):
            if col not in d.columns:
                d[col] = 0
        df = pd.concat([df, d[["embarked_S", "embarked_C", "embarked_Q"]]], axis=1)
    if "family_size" not in df.columns and all(c in df.columns for c in ("sibsp", "parch")):
        df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
    if "fare_per_person" not in df.columns and "fare" in df.columns:
        if "family_size" in df.columns:
            df["fare_per_person"] = df["fare"].fillna(0) / df["family_size"].clip(lower=1)
        else:
            df["fare_per_person"] = df["fare"].fillna(0)
    if "age_group" not in df.columns and "age" in df.columns:
        bins = [0, 12, 18, 35, 60, 120]
        labels = ["child", "teen", "young_adult", "adult", "senior"]
        df["age_group"] = pd.cut(df["age"].fillna(-1), bins=bins, labels=labels, include_lowest=True)
        df["age_group"] = df["age_group"].astype("category").cat.codes
    for col in list(df.columns):
        if df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes
    return df


def normalize_params(params: dict) -> dict:
    p = params.copy()
    if "lambda_l2" in p:
        p["reg_lambda"] = p.pop("lambda_l2")
    if "min_data_in_leaf" in p:
        p["min_child_samples"] = p.pop("min_data_in_leaf")
    p.setdefault("verbosity", -1)
    p.setdefault("n_jobs", -1)
    return p


def parse_optuna_trials(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Try columns like 'value' and 'params_*'
    param_cols = [c for c in df.columns if c.startswith("params_")]
    if "value" in df.columns and param_cols:
        trials = []
        for _, row in df.iterrows():
            params = {}
            for pc in param_cols:
                k = pc.replace("params_", "")
                v = row[pc]
                # try to literal_eval numbers/booleans
                try:
                    v2 = ast.literal_eval(str(v))
                except Exception:
                    v2 = v
                params[k] = v2
            trials.append({"value": row["value"], "params": params})
        return trials
    # fallback: maybe file contains JSON in a column
    try:
        if "params" in df.columns:
            trials = []
            for _, r in df.iterrows():
                p = r["params"]
                if isinstance(p, str):
                    p = json.loads(p)
                trials.append({"value": r.get("value", None), "params": p})
            return trials
    except Exception:
        return None
    return None


def main(top_k: int = 5):
    script_dir = Path(__file__).resolve().parent
    train, test = load_data(script_dir)
    train = ensure_features(train)
    test = ensure_features(test)

    features = [c for c in train.columns if c not in ("survived", "id", "PassengerId")]
    X = train[features].copy()
    y = train["survived"].copy()

    trials = parse_optuna_trials(script_dir / "optuna_trials.csv")
    if trials is None:
        print("optuna_trials.csv not found or unreadable; using fallback grid (K may be smaller)")
        trials = [
            {"value": None, "params": {"learning_rate": 0.01, "n_estimators": 1200, "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 30, "lambda_l2": 0.3}},
            {"value": None, "params": {"learning_rate": 0.03, "n_estimators": 800, "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 25, "lambda_l2": 0.2}},
            {"value": None, "params": {"learning_rate": 0.05, "n_estimators": 600, "num_leaves": 31, "max_depth": 4, "min_data_in_leaf": 20, "lambda_l2": 0.1}},
        ]

    # sort by value descending if value present
    trials_with_value = [t for t in trials if t.get("value") is not None]
    if trials_with_value:
        trials_sorted = sorted(trials, key=lambda t: float(t["value"]), reverse=True)
    else:
        trials_sorted = trials

    top = trials_sorted[:top_k]
    print(f"Using top {len(top)} param sets for ensemble")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_val_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        preds_stack = []
        for t in top:
            params = normalize_params(t["params"].copy())
            model = LGBMClassifier(random_state=42, **params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc")
            preds = model.predict_proba(X_va)[:, 1]
            preds_stack.append(preds)
        # average predictions
        avg_preds = np.mean(np.vstack(preds_stack), axis=0)
        score = roc_auc_score(y_va, avg_preds)
        ensemble_val_scores.append(score)
        print(f"Fold {fold_idx} ensemble AUC: {score:.6f}")

    print(f"Ensemble CV mean AUC: {np.mean(ensemble_val_scores):.6f} (std {np.std(ensemble_val_scores):.6f})")

    # Train full models and average predictions for submission
    preds_full = []
    for t in top:
        params = normalize_params(t["params"].copy())
        model = LGBMClassifier(random_state=42, **params)
        model.fit(X, y)
        preds_full.append(model.predict_proba(test[features])[:, 1])
    avg_test_preds = np.mean(np.vstack(preds_full), axis=0)

    if "id" in test.columns:
        ids = test["id"]
    elif "PassengerId" in test.columns:
        ids = test["PassengerId"]
    else:
        ids = test.index.to_series().reset_index(drop=True)

    sub = pd.DataFrame({"id": ids.values, "survived": avg_test_preds})
    out_path = script_dir / "submission_ensemble_topk.csv"
    sub.to_csv(out_path, header=False, index=False)
    print("Wrote ensemble submission to:", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    main(top_k=args.top_k)
