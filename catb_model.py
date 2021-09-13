
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.metrics import roc_auc_score
import seaborn as sns

class Cat_model:
    def __init__(self):
        self.params = {

            # 固定パラメータ
            'early_stopping_rounds' : 60,
            'iterations' : 300, 
            'eval_metric':"AUC",
            'random_seed' :42,
            'learning_rate' : 0.05,
            'verbose': 10,
            'cat_features': categories,

            # 可変パラメータ
            'depth': 3,
            'l2_leaf_reg': 19,
            'border_count': 72,
            'random_strength': 40,
            'bagging_temperature': 1.4297724879798401
        }
        self.model = []# 学習済みモデル保存

    def train(self, train_X, train_Y, categories, has_weights=False, show_importance=False):

        if has_weights:
            pass

        # 性能評価用およびモデル保存用の変数を宣言
        folds = 5
        fold_count = np.zeros(5)# foldごとのモデル数
        seed_count = np.zeros(16)# random seedごとのモデル数
        all_aucs = []# スコアを保存(シード単位)
        ind_aucs = []# スコアを保存(1モデル単位)
        ind_train_aucs = []# trainデータに対するスコアを保存(可視化用)

        for s in range(16):

            # 特定のseedからパラメータに変化を加える
            if s == 9:
                print("========================  parame update... ======================== ")
                cat_params["l2_leaf_reg"]=25
                cat_params["learning_rate"]=0.08
                cat_params["early_stopping_rounds"]=45
            elif s == 12:
                print("========================  parame update... ======================== ")
                cat_params={

                    # 固定パラ
                    'early_stopping_rounds' : 60,
                    'iterations' : 300, 
                    'eval_metric':"AUC",
                    'random_seed' :42,
                    'learning_rate' : 0.05,
                    'verbose': 10,
                    'cat_features': categories,

                    # 可変パラ
                    'depth': 3,
                    'l2_leaf_reg': 14,
                    'border_count': 100,
                    'random_strength': 61,
                    'bagging_temperature': 1.8854097488371255
                }

            aucs = []
            self.params['random_seed'] = s
            kf=KFold(n_splits=folds, shuffle=True, random_state=s)

            for i, (train_index, val_index) in enumerate(kf.split(train_X, train_Y)):

                # trainデータ, valデータ分割
                X_train = train_X.iloc[train_index]
                y_train = train_Y.iloc[train_index]
                X_valid = train_X.iloc[val_index]
                y_valid = train_Y.iloc[val_index]

                # catboost用データへ変形
                train_pool = Pool(X_train, y_train)
                eval_pool = Pool(X_valid, y_valid)

                # model定義, 学習        
                model_cat=CatBoostClassifier(**cat_params)
                model_cat.fit(train_pool, eval_set=eval_pool, use_best_model=True,plot=False)

                # 重要度の可視化(optinal)
                if show_importance:
                    feature_importance = model_cat.get_feature_importance()
                    plt.figure(figsize=(12, 4))
                    plt.barh(range(len(feature_importance)),
                            feature_importance,
                            tick_label=dataset.feature_names)

                    plt.xlabel('importance')
                    plt.ylabel('features')
                    plt.grid()
                    plt.show()

                # train dataに対する予測
                y_train_pred = model_cat.predict_proba(X_train)[:,1]
                train_score = roc_auc_score(train_Y, y_train_pred)
                ind_train_aucs.append(train_score)
                print("valid roc_auc_score: ", train_score)

                # valid dataに対する予測
                y_pred = model_cat.predict_proba(X_valid)[:,1]
                score = roc_auc_score(y_valid, y_pred)
                ind_aucs.append(score)# 全てのモデルのscoreを保存
                print("valid roc_auc_score: ", score)

                # モデルの保存(train score及びvalid scoreが0.7以上のモデルのみ) << データが著しく不均衡に配分され, 学習できたかったモデルを弾くため
                if (train_score >= 0.7) and  (score >= 0.7):
                    self.models.append(model_lgb)
                    aucs.append(score)
                    fold_count[i] += 1
                    seed_count[s] += 1

            # seedごとのスコアを表示
            if not np.isnan(np.mean(aucs)):
                seed_score = np.mean(aucs)
                print("seed:",s, " mean_auc_score:", seed_score)
                all_aucs.append(seed_score)
            else:
                print("[Warning] seed:",s, " {none model in this seed.}")

        print("========================== 学習終了 ============================")

        # 全体の学習結果を表示
        print("model_count(n>=0.7): ",np.sum(fold_count))
        print("fold_count:",fold_count)
        print("seed_count:",seed_count)
        print("train_mean_all_aucs", np.mean(ind_train_aucs))
        print("mean_all_aucs:", np.mean(all_aucs))

        # valid dataに対するscoreの分布を表示
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        plt.hist(ind_aucs, bins=30)
        plt.show()

        # train dataに対するscoreの分布を表示
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        plt.hist(ind_train_aucs, bins=30)
        plt.show()
    
    def predict(self, test_X):

        preds = []
        for i, model in enumerate(self.models):
            pred = model.predict_proba(test_X)[:, 1]
            preds.append(pred)
        print("num of models: ", i+1)
        
        preds_array=np.array(preds)
        cat_pred=np.mean(preds_array, axis=0)

        plt.plot(range(cat_pred.shape[0]),cat_pred)
        plt.show()

        return cat_pred
                
    # def param_tuning():
    # パラメータチューニングのコードは省略(optunaでチューニング)

def cat_pre_categorical(all_df, categories):

    for i,col in enumerate(all_df.columns):
        if col in categories:
            all_df[col] = all_df[col].astype('int64')
        

if __name__ == "__main__":

    # データの取得表示
    all_df = pd.read_csv("path to all DataFrame file")
    print(all_df.head())
    categories = ['month', 'season', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'comb_fea']

    # catboost用にカテゴリ変数の前処理
    all_df = cat_pre_categorical(all_df, categories)

    # train_test分割
    train_data = all_df[~all_df["y"].isnull()]
    test_data = all_df[all_df["y"].isnull()]

    # 学習データ準備
    train_X = train_data.drop("y",axis=1)
    train_Y = train_data["y"]
    test_X = test_data.drop("y",axis=1)

    # 学習
    model = Cat_model(train_X, train_Y, categories)
    model.train(train_X, train_Y, categories)
    pred = model.predict(test_X)

    # 提出用ファイル作成・保存
    sample_df = pd.read_csv("path to submission sample file",header=None)
    sample_df[1] = pd.DataFrame(pred)
    sample_df.to_csv("submission file name",index=False, header=False)

