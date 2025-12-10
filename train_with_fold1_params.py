"""Find the Optuna trial that gave best AUC on fold index 1, retrain on full data.

Produces `submission_fold1params_fulltrain.csv` in the script directory.
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
        raise FileNotFoundError(path)
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
    # fallback if params column exists
    if "params" in df.columns:
        for _, row in df.iterrows():
            p = row["params"]
            if isinstance(p, str):
                try:
                    p = json.loads(p)
                except Exception:
                    p = ast.literal_eval(p)
            trials.append({"value": row.get("value", None), "params": p})
        return trials
    raise RuntimeError("Unrecognized optuna_trials.csv format")


def main():
    script_dir = Path(__file__).resolve().parent
    train, test = load_data(script_dir)
    train = ensure_features(train)
    test = ensure_features(test)

    features = [c for c in train.columns if c not in ("survived", "id", "PassengerId")]
    X = train[features].copy()
    y = train["survived"].copy()

    trials = parse_optuna_trials(script_dir / "optuna_trials.csv")
    print(f"Loaded {len(trials)} trials from optuna_trials.csv")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate each trial on fold index 1
    fold_idx_target = 1
    best_trial = None
    best_score = -1.0
    for i, t in enumerate(trials):
        params = normalize_params(t["params"].copy())
        # do CV but only compute for fold 1
        for fi, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
            if fi != fold_idx_target:
                continue
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            model = LGBMClassifier(random_state=42, **params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc")
            preds = model.predict_proba(X_va)[:, 1]
            score = roc_auc_score(y_va, preds)
            print(f"Trial {i} fold{fi} AUC: {score:.6f}")
            if score > best_score:
                best_score = score
                best_trial = t

    print(f"Best trial on fold {fold_idx_target}: AUC={best_score:.6f}, params={best_trial['params']}" )

    # Retrain on full data with best params
    final_params = normalize_params(best_trial["params"].copy())
    final_model = LGBMClassifier(random_state=42, **final_params)
    final_model.fit(X, y)
    test_preds = final_model.predict_proba(test[features])[:, 1]

    if "id" in test.columns:
        ids = test["id"]
    elif "PassengerId" in test.columns:
        ids = test["PassengerId"]
    else:
        ids = test.index.to_series().reset_index(drop=True)

    sub = pd.DataFrame({"id": ids.values, "survived": test_preds})
    out_path = script_dir / "submission_fold1params_fulltrain.csv"
    sub.to_csv(out_path, header=False, index=False)
    print("Wrote submission:", out_path)


if __name__ == "__main__":
    main()
