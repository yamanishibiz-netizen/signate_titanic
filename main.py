import os
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# スクリプトファイルのあるディレクトリを基準にデータを読む
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# データ読み込み
train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

# sexを数値化
train["sex"] = train["sex"].map({"male": 0, "female": 1})
test["sex"] = test["sex"].map({"male": 0, "female": 1})

# embarked補完
for df in [train, test]:
    df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# age補完
for df in [train, test]:
    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda x: x.fillna(x.median()))

# fare補完
for df in [train, test]:
    df["fare"] = df.groupby(["pclass", "embarked"])["fare"].transform(lambda x: x.fillna(x.median()))

# embarkedをOne-Hot化
train = pd.get_dummies(train, columns=["embarked"], prefix="embarked")
test = pd.get_dummies(test, columns=["embarked"], prefix="embarked")

# 🔹追加特徴量
train["family_size"] = train["sibsp"] + train["parch"] + 1
test["family_size"] = test["sibsp"] + test["parch"] + 1

train["fare_per_person"] = train["fare"] / train["family_size"]
test["fare_per_person"] = test["fare"] / test["family_size"]

train["is_alone"] = (train["family_size"] == 1).astype(int)
test["is_alone"] = (test["family_size"] == 1).astype(int)

train["age_group"] = pd.cut(train["age"], bins=[0,12,18,60,100], labels=[0,1,2,3])
test["age_group"] = pd.cut(test["age"], bins=[0,12,18,60,100], labels=[0,1,2,3])

# age_group を数値化して欠損を埋める
train["age_group"] = train["age_group"].astype(float).fillna(-1).astype(int)
test["age_group"] = test["age_group"].astype(float).fillna(-1).astype(int)

# 敬称抽出（`name` 列がある場合のみ抽出、なければプレースホルダを作成）
if "name" in train.columns and "name" in test.columns:
    train["title"] = train["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    test["title"] = test["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    for df in [train, test]:
        df["title"] = df["title"].replace(["Mlle","Ms"], "Miss").replace(["Mme"], "Mrs")
        df["title"] = df["title"].replace(
            ["Dr","Rev","Col","Major","Capt","Countess","Lady","Sir","Jonkheer","Don"], "Rare"
        )
else:
    # `name` がないデータセット向けに汎用ラベルを付与してダミー化可能にする
    train["title"] = "NoName"
    test["title"] = "NoName"

train = pd.get_dummies(train, columns=["title"], prefix="title")
test = pd.get_dummies(test, columns=["title"], prefix="title")

# キャビン頭文字（`cabin` 列がある場合は先頭文字を使い、無ければ 'U' を使う）
if "cabin" in train.columns:
    train["cabin_initial"] = train["cabin"].fillna("U").str[0]
else:
    train["cabin_initial"] = "U"

if "cabin" in test.columns:
    test["cabin_initial"] = test["cabin"].fillna("U").str[0]
else:
    test["cabin_initial"] = "U"

train = pd.get_dummies(train, columns=["cabin_initial"], prefix="cabin")
test = pd.get_dummies(test, columns=["cabin_initial"], prefix="cabin")

# --- 重要: train と test のダミー列を揃える ---
# train にあって test にない列は 0 を埋める。逆も同様に行う。
for col in train.columns:
    if col not in test.columns and col not in ["survived", "name", "ticket", "cabin"]:
        test[col] = 0
for col in test.columns:
    if col not in train.columns:
        # survived は train 側にしかないため追加は不要だが
        # モデル学習や列揃えのため 0 を埋める
        train[col] = 0

# 特徴量選択
features = [col for col in train.columns if col not in ["survived","name","ticket","cabin"]]
X = train[features]
y = train["survived"]
X_test = test[features]

# train/valid分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBMモデル
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc")

# 予測
preds = model.predict_proba(X_test)[:,1]

# 提出ファイル作成（ヘッダなし2列）
# テストデータのID列名はデータセットにより異なるためフォールバックを用意
id_col = "id" if "id" in test.columns else ("PassengerId" if "PassengerId" in test.columns else None)
if id_col is None:
    # 最後の手段として最初の列をID扱い
    id_col = test.columns[0]

submission = pd.DataFrame({"id": test[id_col], "survived": preds})
submission.to_csv(os.path.join(BASE_DIR, "submission.csv"), index=False, header=False)
