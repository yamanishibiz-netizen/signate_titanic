"""Evaluate the Optuna trial that was best on fold1 using full 5-fold CV.

Prints fold AUCs and mean/std.
"""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier


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
    if train is None:
        raise RuntimeError("train not found in script dir")
    return train


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
    df = pd.read_csv(path)
    param_cols = [c for c in df.columns if c.startswith("params_")]
    trials = []
    if "value" in df.columns and param_cols:
        for _, row in df.iterrows():
            params = {}
            for pc in param_cols:
                k = pc.replace("params_", "")
                v = row[pc]
                try:
                    v2 = ast.literal_eval(str(v))
                except Exception:
                    v2 = v
                params[k] = v2
            trials.append({"value": row["value"], "params": params})
        return trials
    raise RuntimeError("optuna_trials.csv format not recognized")


def main():
    script_dir = Path(__file__).resolve().parent
    train = load_data(script_dir)
    train = ensure_features(train)

    features = [c for c in train.columns if c not in ("survived", "id", "PassengerId")]
    X = train[features].copy()
    y = train["survived"].copy()

    trials = parse_optuna_trials(script_dir / "optuna_trials.csv")

    # Find trial with max fold1 AUC (recompute per trial fold1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_target = 1
    best = None
    best_score = -1.0
    for i, t in enumerate(trials):
        params = normalize_params(t["params"].copy())
        for fi, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
            if fi != fold_target:
                continue
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            model = LGBMClassifier(random_state=42, **params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc")
            auc = roc_auc_score(y_va, model.predict_proba(X_va)[:,1])
            if auc > best_score:
                best_score = auc
                best = t

    print("Fold1-best AUC:", best_score)
    print("Params:", best["params"])

    # Now evaluate this params on full 5-fold CV
    params = normalize_params(best["params"].copy())
    fold_scores = []
    for fi, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = LGBMClassifier(random_state=42, **params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc")
        auc = roc_auc_score(y_va, model.predict_proba(X_va)[:,1])
        fold_scores.append(auc)
        print(f"Fold {fi} AUC: {auc:.6f}")

    print(f"Mean AUC: {np.mean(fold_scores):.6f}, Std: {np.std(fold_scores):.6f}")


if __name__ == "__main__":
    main()
