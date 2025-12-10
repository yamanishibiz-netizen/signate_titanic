import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

script_dir = Path(__file__).resolve().parent

# load train

def load_candidate(name_stems):
    for stem in name_stems:
        for ext in (".csv", ".parquet", ".pkl"):
            p = script_dir / (stem + ext)
            if p.exists():
                if ext == ".csv":
                    return pd.read_csv(p)
                elif ext == ".parquet":
                    return pd.read_parquet(p)
                else:
                    return pd.read_pickle(p)
    return None

train = load_candidate(["train", "train_data", "train_df"]) 
if train is None:
    raise SystemExit("train not found")

# replicate _ensure_features from cv_tuning.py (minimal)
def _ensure_features(df):
    df = df.copy()
    if "embarked" in df.columns:
        d = pd.get_dummies(df["embarked"], prefix="embarked")
        for col in ("embarked_S","embarked_C","embarked_Q"):
            if col not in d.columns:
                d[col] = 0
        df = pd.concat([df, d[["embarked_S","embarked_C","embarked_Q"]]], axis=1)
    if "family_size" not in df.columns and all(c in df.columns for c in ("sibsp","parch")):
        df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
    if "fare_per_person" not in df.columns and "fare" in df.columns:
        if "family_size" in df.columns:
            df["fare_per_person"] = df["fare"].fillna(0) / df["family_size"].clip(lower=1)
        else:
            df["fare_per_person"] = df["fare"].fillna(0)
    if "age_group" not in df.columns and "age" in df.columns:
        bins = [0,12,18,35,60,120]
        labels = ["child","teen","young_adult","adult","senior"]
        df["age_group"] = pd.cut(df["age"].fillna(-1), bins=bins, labels=labels, include_lowest=True)
        df["age_group"] = df["age_group"].astype("category").cat.codes
    # convert core object cols
    core_features = [
        "pclass","sex","age","sibsp","parch","fare",
        "embarked_S","embarked_C","embarked_Q",
        "family_size","fare_per_person","age_group"
    ]
    for col in list(df.columns):
        if col in core_features and df[col].dtype == object:
            df[col] = df[col].astype('category').cat.codes
    if "family_size" in df.columns and "is_alone" not in df.columns:
        df["is_alone"] = (df["family_size"] == 1).astype(int)
    if "fare" in df.columns and "fare_log" not in df.columns:
        df["fare_log"] = np.log1p(df["fare"].fillna(0))
    if "age" in df.columns:
        if "age_na" not in df.columns:
            df["age_na"] = df["age"].isna().astype(int)
        if df["age"].isna().any():
            median_age = df["age"].median()
            df["age"] = df["age"].fillna(median_age)
    if "Name" in df.columns and "title" not in df.columns:
        titles = df["Name"].astype(str).str.extract(r",\s*([^\.]+)\.")
        df["title"] = titles[0].fillna("Unknown").str.strip()
        common = df["title"].value_counts().nlargest(10).index
        df["title"] = df["title"].where(df["title"].isin(common), other="Rare")
        df["title"] = df["title"].astype("category").cat.codes
    return df

train = _ensure_features(train)

available_features = [f for f in [
    "pclass","sex","age","sibsp","parch","fare",
    "embarked_S","embarked_C","embarked_Q",
    "family_size","fare_per_person","age_group"] if f in train.columns]
for extra in ("is_alone","fare_log","age_na","title"):
    if extra in train.columns and extra not in available_features:
        available_features.append(extra)

X = train[available_features].copy()
y = train["survived"].copy()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# grid best params (from cv_tuning.py)
grid_best = {"learning_rate":0.03, "n_estimators":800, "num_leaves":31, "max_depth":5, "min_data_in_leaf":25, "lambda_l2":0.2}

def normalize_params(params):
    p = params.copy()
    if "lambda_l2" in p:
        p["reg_lambda"] = p.pop("lambda_l2")
    if "min_data_in_leaf" in p:
        p["min_child_samples"] = p.pop("min_data_in_leaf")
    p.setdefault("verbosity", -1)
    p.setdefault("n_jobs", -1)
    return p

# load optuna_trials.csv and extract best row
opt_csv = script_dir / "optuna_trials.csv"
opt_best = None
if opt_csv.exists():
    df = pd.read_csv(opt_csv)
    if 'value' in df.columns:
        best_idx = df['value'].idxmax()
        # collect params prefixed with 'params_'
        params = {}
        for c in df.columns:
            if c.startswith('params_'):
                k = c.replace('params_', '')
                params[k] = df.at[best_idx, c]
        opt_best = params

# evaluate function
from sklearn.metrics import roc_auc_score

def eval_params(params, name="params"):
    p = normalize_params(params)
    scores = []
    for tr_idx, va_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = LGBMClassifier(random_state=42, **p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[ ] )
        preds = model.predict_proba(X_va)[:,1]
        scores.append(roc_auc_score(y_va, preds))
    print(f"{name} -> fold AUCs: {scores} mean: {np.mean(scores):.6f}")
    return np.mean(scores)

print('Available features:', available_features)
print('Training rows:', X.shape)
print('\nEvaluating grid-best (original) params:')
eval_params(grid_best, 'grid_best')
if opt_best is not None:
    # convert strings to numeric where possible
    for k,v in list(opt_best.items()):
        try:
            opt_best[k] = float(v)
            if opt_best[k].is_integer():
                opt_best[k] = int(opt_best[k])
        except Exception:
            pass
    print('\nEvaluating Optuna best from optuna_trials.csv:')
    eval_params(opt_best, 'optuna_best')
else:
    print('No optuna_trials.csv found, or no value column.')

# show submission sample
sub = script_dir / 'submission.csv'
if sub.exists():
    print('\nsubmission.csv sample:')
    print(pd.read_csv(sub, header=None).head(10))
else:
    print('\nsubmission.csv not found')
