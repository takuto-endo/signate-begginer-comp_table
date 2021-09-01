import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

class lgb_model:
    def __init__(self):
        self.num_boost_round = 1000
        self.params = {

            # 固定パラメータ
            "objective":"binary",
            "metric":"auc",
            "learning_rate":0.1,
            "random_seed":1,
            "verbosity": 0,
            "drop_rate": 0.1,
            "force_col_wise":True,

            # 可変パラメータ
            'num_leaves': 11,
            'max_bin': 53,
            'bagging_fraction': 0.41310549343885966,
            'bagging_freq': 1,
            'feature_fraction': 0.5231055338015229,
            'min_data_in_leaf': 14,
            'min_sum_hessian_in_leaf': 6
        }
        self.models = []# 学習済みのモデルを保存

    def train(self, train_X, train_Y, categories, has_weights=False, show_importance=False):

        if has_weights:
            weight_list = np.ones(train_Y.shape[0])
            for i in range(len(train_Y)):
                if train_Y[i] == 1:
                    weight_list[i] = 5
            self.params["weight_list"] = weight_list

        # 性能評価用およびモデル保存用の変数を宣言
        folds=5
        fold_count = np.zeros(5)# foldごとのモデル数
        seed_count = np.zeros(16)# random seedごとのモデル数
        all_aucs = []# スコアを保存(シード単位)
        ind_aucs = []# スコアを保存(1モデル単位)
        ind_train_aucs = []# trainデータに対するスコアを保存(可視化用)
        
        # 複数のランダムシードで学習したモデルをバギングすることで汎化性能up
        for s in range(16):

            aucs = []# ランダムシードごとのスコア保存
            self.params["random_seed"] = s
            kf=KFold(n_splits=folds, shuffle=True, random_state=s)
            for i, (train_index, val_index) in enumerate(kf.split(train_X)):

                # trainデータ, validデータ分割
                X_train = train_X.iloc[train_index]
                y_train = train_Y.iloc[train_index]
                X_valid = train_X.iloc[val_index]
                y_valid = train_Y.iloc[val_index]

                # lightgbm用にデータを変形
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
                
                # モデルの学習
                model_lgb = lgb.train(self.params,
                                    lgb_train,
                                    valid_sets = lgb_eval,
                                    num_boost_round = self.num_boost_round,
                                    early_stopping_rounds = 100,
                                    categorical_feature  = categories,
                                    verbose_eval = 10)
                

                # train dataに対する予測
                y_train_pred = model_lgb.predict(train_X,num_iteration=model_lgb.best_iteration)
                train_score = roc_auc_score(train_Y, y_train_pred)
                ind_train_aucs.append(train_score)
                print("valid roc_auc_score: ", train_score)

                # valid dataに対する予測
                y_pred = model_lgb.predict(X_valid,num_iteration=model_lgb.best_iteration)
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

        # 重要度を表示(optional)
        if show_importance:
            for i in range(10):
                lgb.plot_importance(self.models[i], importance_type="gain", max_num_features=15)

        # train dataに対するscoreの分布を表示
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        plt.hist(ind_train_aucs, bins=30)
        plt.show()

    def predict(self, test_X):
        
        preds = []
        for i,model in enumerate(self.models):
            pred = model.predict(test_X, num_iteration=model.best_iteration)
            preds.append(pred)
        print("num of models: ",i+1)

        preds_array=np.array(preds)
        lgbm_pred=np.mean(preds_array, axis=0)

        plt.plot(range(lgbm_pred.shape[0]),lgbm_pred)
        plt.show()

        return lgbm_pred

    # def param_tuning():
    # パラメータチューニングのコードは省略(optunaでチューニング)

if __name__ == '__main__':

    # データの取得・表示
    all_df = pd.read_csv("../input/all_df.csv")
    print(all_df.head())

    # rain_test分割
    train_data = all_df[~all_df["y"].isnull()]
    test_data = all_df[all_df["y"].isnull()]

    # 学習データ準備
    train_X = train_data.drop("y",axis=1)
    train_Y = train_data["y"]
    test_X = test_data.drop("y",axis=1)
    categories = ['month', 'season', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'comb_fea']

    # 学習
    model = lgb_model()
    model.train(train_X, train_Y, categories, has_weights=True, show_importance=True)
    pred = model.predict(test_X)

    # 提出用ファイル作成・保存
    sample_df = pd.read_csv("path to submission sample file",header=None)
    sample_df[1]=pd.DataFrame(pred)
    sample_df.to_csv("submission file name",index=False, header=False)