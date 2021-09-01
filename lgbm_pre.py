
import numpy as np
import pandas as pd

# ============================ データ読み込み ============================
train_df = pd.read_csv("path to train data")
test_df = pd.read_csv("path to test data")
all_df = pd.concat([train_df, test_df])

# ============================ 前処理 ============================

categories = []# カテゴリ変数保存 lightgbm,catboost学習時に指定するため最後に表示


# 1.学習に必要ない列削除 (balanceは全てのデータが0)
all_df = all_df.drop(["id","balance"],axis=1)

# 2.月データを数字に変換 (単にLabel Encodingすると順序がバラバラになる可能性あり)
#   NN用に変換する場合は1月と12月の繋がりが表現できないため注意(決定木のモデルなら複数の分岐で表現可能)
month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
all_df['month'] = all_df['month'].map(month_mapping)
categories.append('month')

# 3.季節変数追加
season_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}
all_df['season'] = all_df['month'].map(season_mapping)
categories.append('season')

# 4.組み合わせ変数の作成
#   全てのカテゴリ変数の組み合わせ >> 何個かのグループに分けることも試したが全ての組み合わせが一番精度が向上した
count = 0
feature_list = ["default", "loan", "contact", "education",'job', 'marital', 'housing', 'campaign', 'poutcome']
for c in feature_list:
    if count == 0:
        all_df["comb_fea"] = all_df[c].astype(str)
        count += 1
    else:
        all_df["comb_fea"] += "_" + all_df[c].astype(str)
feature_list += ["comb_fea"]

# 5.カテゴリ変数の基礎統計量(今回は頻度のみ)
for c in feature_list:
    d = all_df[c].value_counts().to_dict()
    all_df['%s_count'%c] = all_df[c].apply(lambda  x: d.get(x,0))

# 6.object型のデータをLabelEncoding
from sklearn.preprocessing import LabelEncoder
for col in all_df.columns:
    le = LabelEncoder()
    if all_df[col].dtypes == "object":
        le = le.fit(all_df[col])
        all_df[col]= le.transform(all_df[col])
        categories.append(col)

# ============================ 表示および保存 ============================

print(all_df)
print(categories)
all_df.to_csv("save filename",index=False)

