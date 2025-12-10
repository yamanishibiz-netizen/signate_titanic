import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# train / test がグローバルに無ければ csv または parquet を同ディレクトリから探す
try:
    train  # type: ignore
    test   # type: ignore
except NameError:
    script_dir = Path(__file__).resolve().parent
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
    test = load_candidate(["test", "test_data", "test_df"])

    if train is None or test is None:
        raise RuntimeError(
            "train/test が見つかりません。スクリプトの同ディレクトリに train.csv/test.csv などを置くか、"
            "先に train/test DataFrame を定義してください。"
        )

# 残す特徴のみ選択
core_features = [
    "pclass","sex","age","sibsp","parch","fare",
    "embarked_S","embarked_C","embarked_Q",
    "family_size","fare_per_person","age_group"
]
def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # embarked one-hot (embarked_S, embarked_C, embarked_Q)
    if "embarked" in df.columns:
        d = pd.get_dummies(df["embarked"], prefix="embarked")
        for col in ("embarked_S", "embarked_C", "embarked_Q"):
            if col not in d.columns:
                d[col] = 0
        df = pd.concat([df, d[["embarked_S", "embarked_C", "embarked_Q"]]], axis=1)

    # family_size = sibsp + parch + 1
    if "family_size" not in df.columns and all(c in df.columns for c in ("sibsp", "parch")):
        df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1

    # fare_per_person = fare / family_size (fallback to fare if family_size missing)
    if "fare_per_person" not in df.columns and "fare" in df.columns:
        if "family_size" in df.columns:
            df["fare_per_person"] = df["fare"].fillna(0) / df["family_size"].clip(lower=1)
        else:
            df["fare_per_person"] = df["fare"].fillna(0)

    # age_group: simple bucketing from age
    if "age_group" not in df.columns and "age" in df.columns:
        bins = [0, 12, 18, 35, 60, 120]
        labels = ["child", "teen", "young_adult", "adult", "senior"]
        df["age_group"] = pd.cut(df["age"].fillna(-1), bins=bins, labels=labels, include_lowest=True)
        # convert to numeric codes so models won't choke on strings
        df["age_group"] = df["age_group"].astype("category").cat.codes

    # convert any remaining object dtype core features to categorical codes
    for col in list(df.columns):
        if col in core_features and df[col].dtype == object:
            df[col] = df[col].astype('category').cat.codes

    return df


# try to create missing engineered features in both train/test when possible
train = _ensure_features(train)
test = _ensure_features(test)

# choose only features that actually exist
available_features = [f for f in core_features if f in train.columns]
missing = [f for f in core_features if f not in train.columns]
if missing:
    print("Warning: following core features missing from train and will be skipped:", missing)

X = train[available_features].copy()
y = train["survived"].copy()
X_test = test[available_features].copy() if all(f in test.columns for f in available_features) else None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 軽いグリッド（組み合わせは最小限）
param_grid = [
    {"learning_rate":0.03, "n_estimators":800, "num_leaves":31, "max_depth":5, "min_data_in_leaf":25, "lambda_l2":0.2},
    {"learning_rate":0.01, "n_estimators":1200, "num_leaves":15, "max_depth":5, "min_data_in_leaf":30, "lambda_l2":0.3},
    {"learning_rate":0.05, "n_estimators":600, "num_leaves":31, "max_depth":4, "min_data_in_leaf":20, "lambda_l2":0.1},
]

def normalize_params(params):
    p = params.copy()
    # LightGBM sklearn API でのパラメータ名に合わせる
    if "lambda_l2" in p:
        p["reg_lambda"] = p.pop("lambda_l2")
    if "min_data_in_leaf" in p:
        p["min_child_samples"] = p.pop("min_data_in_leaf")
    # reduce logging noise and use all cores
    p.setdefault("verbosity", -1)
    p.setdefault("n_jobs", -1)
    return p

results = []
for params in param_grid:
    params_norm = normalize_params(params)
    scores = []
    for tr_idx, va_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = LGBMClassifier(
            random_state=42,
            **params_norm
        )
        # verbose を False にして出力を抑制
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc")
        preds_va = model.predict_proba(X_va)[:,1]
        scores.append(roc_auc_score(y_va, preds_va))
    results.append((params, np.mean(scores)))

# ベストパラメータ選択
best_params, best_cv = sorted(results, key=lambda x: x[1], reverse=True)[0]
print("Best CV AUC:", best_cv, "Params:", best_params)

# ベストで全学習→テスト予測
best_params_norm = normalize_params(best_params)
best_model = LGBMClassifier(random_state=42, **best_params_norm)
best_model.fit(X, y)
if X_test is not None:
    test_preds = best_model.predict_proba(X_test)[:,1]
    # フォーマット要件: 1列目=id, 2列目=生存確率、ヘッダ無しのCSV
    if "id" in test.columns:
        ids = test["id"]
    elif "PassengerId" in test.columns:
        ids = test["PassengerId"]
    else:
        # 最悪、インデックスを id として使う
        ids = test.index.to_series().reset_index(drop=True)

    sub = pd.DataFrame({"id": ids.values, "survived": test_preds})
    out_path = script_dir / "submission.csv"
    # ヘッダ無しで保存（評価フォーマット準拠）
    sub.to_csv(out_path, header=False, index=False)
    print(f"Saved predictions to: {out_path} (headerless id,prob)")
else:
    test_preds = None