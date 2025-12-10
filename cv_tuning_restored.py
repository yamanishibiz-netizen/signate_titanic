import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ---- data loading ----
try:
    train  # type: ignore
    test   # type: ignore
except NameError:
    script_dir = Path(__file__).resolve().parent

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
        raise RuntimeError("train/test が見つかりません。CSV/Parquet を同ディレクトリに置いてください。")


# ---- feature helpers ----
core_features = [
    "pclass", "sex", "age", "sibsp", "parch", "fare",
    "embarked_S", "embarked_C", "embarked_Q",
    "family_size", "fare_per_person", "age_group",
]


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
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
        if col in core_features and df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes

    return df


# ---- prepare data ----
train = _ensure_features(train)
test = _ensure_features(test)

available_features = [f for f in core_features if f in train.columns]
missing = [f for f in core_features if f not in train.columns]
if missing:
    print("Warning: missing core features, skipping:", missing)

X = train[available_features].copy()
y = train["survived"].copy()
X_test = test[available_features].copy() if all(f in test.columns for f in available_features) else None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ---- simple grid and CV ----
param_grid = [
    {"learning_rate": 0.03, "n_estimators": 800, "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 25, "lambda_l2": 0.2},
    {"learning_rate": 0.01, "n_estimators": 1200, "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 30, "lambda_l2": 0.3},
    {"learning_rate": 0.05, "n_estimators": 600, "num_leaves": 31, "max_depth": 4, "min_data_in_leaf": 20, "lambda_l2": 0.1},
]


def normalize_params(params: dict) -> dict:
    p = params.copy()
    if "lambda_l2" in p:
        p["reg_lambda"] = p.pop("lambda_l2")
    if "min_data_in_leaf" in p:
        p["min_child_samples"] = p.pop("min_data_in_leaf")
    p.setdefault("verbosity", -1)
    p.setdefault("n_jobs", -1)
    return p


results = []
for params in param_grid:
    pnorm = normalize_params(params)
    scores = []
    for tr_idx, va_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = LGBMClassifier(random_state=42, **pnorm)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc")
        preds = model.predict_proba(X_va)[:, 1]
        scores.append(roc_auc_score(y_va, preds))
    results.append((params, float(np.mean(scores))))

best_params, best_cv = sorted(results, key=lambda x: x[1], reverse=True)[0]
print("Best CV AUC:", best_cv, "Params:", best_params)


# ---- train on full and predict ----
best_model = LGBMClassifier(random_state=42, **normalize_params(best_params))
best_model.fit(X, y)

if X_test is not None:
    preds_test = best_model.predict_proba(X_test)[:, 1]
    if "id" in test.columns:
        ids = test["id"]
    elif "PassengerId" in test.columns:
        ids = test["PassengerId"]
    else:
        ids = test.index.to_series().reset_index(drop=True)
    sub = pd.DataFrame({"id": ids.values, "survived": preds_test})
    out_path = script_dir / "submission.csv"
    sub.to_csv(out_path, header=False, index=False)
    print("Saved predictions to:", out_path)
