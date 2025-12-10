import numpy as np
import pandas as pd
import os
from pathlib import Path
from lightgbm import LGBMClassifier

# スクリプトファイルのあるディレクトリを基準にデータを読む
BASE_DIR = Path(__file__).resolve().parent
train = pd.read_csv(BASE_DIR / "train.csv")
test = pd.read_csv(BASE_DIR / "test.csv")

# 残す特徴のみ選択
core_features = [
    "pclass","sex","age","sibsp","parch","fare",
    "embarked_S","embarked_C","embarked_Q",
    "family_size","fare_per_person","age_group"
]

def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # embarked one-hot
    if "embarked" in df.columns:
        d = pd.get_dummies(df["embarked"], prefix="embarked")
        for col in ("embarked_S", "embarked_C", "embarked_Q"):
            if col not in d.columns:
                d[col] = 0
        df = pd.concat([df, d[["embarked_S","embarked_C","embarked_Q"]]], axis=1)

    # family_size
    if "family_size" not in df.columns and all(c in df.columns for c in ("sibsp","parch")):
        df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1

    # fare_per_person
    if "fare_per_person" not in df.columns and "fare" in df.columns:
        if "family_size" in df.columns:
            df["fare_per_person"] = df["fare"].fillna(0) / df["family_size"].clip(lower=1)
        else:
            df["fare_per_person"] = df["fare"].fillna(0)

    # age_group
    if "age_group" not in df.columns and "age" in df.columns:
        bins = [0,12,18,35,60,120]
        labels = ["child","teen","young_adult","adult","senior"]
        df["age_group"] = pd.cut(df["age"].fillna(-1), bins=bins, labels=labels, include_lowest=True)
        df["age_group"] = df["age_group"].astype("category").cat.codes

    # object型をカテゴリコード化
    for col in list(df.columns):
        if col in core_features and df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes

    return df

# 特徴量生成
train = _ensure_features(train)
test = _ensure_features(test)

# 使用する特徴量
features = [f for f in core_features if f in train.columns]
X = train[features].copy()
y = train["survived"].copy()
X_test = test[features].copy()

# CVで得られたベストパラメータを反映
best_params = {
    "learning_rate":0.03,
    "n_estimators":800,
    "num_leaves":31,
    "max_depth":5,
    "min_child_samples":25,
    "reg_lambda":0.2,
    "verbosity":-1,
    "n_jobs":-1,
    "random_state":42
}

# モデル学習
model = LGBMClassifier(**best_params)
model.fit(X, y)

# 予測
preds = model.predict_proba(X_test)[:,1]

# 提出ファイル作成
if "id" in test.columns:
    ids = test["id"]
elif "PassengerId" in test.columns:
    ids = test["PassengerId"]
else:
    ids = test.index.to_series().reset_index(drop=True)

submission = pd.DataFrame({"id": ids.values, "survived": preds})
submission.to_csv(BASE_DIR / "submission.csv", header=False, index=False)
print("Saved submission.csv")
