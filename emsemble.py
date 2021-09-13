
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


pred1 = pd.read_csv("csv file path 1", header=None)[1]
pred2 = pd.read_csv("csv file path 2", header=None)[1]

# 底を0に揃える
pred1 -= np.min(pred1)
pred2 -= np.min(pred2)

# グラフで見ながら割合を調整
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(range(1500),pred1*2, label="pred1")
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(1500),pred2, label="pred2")
plt.legend()
plt.show()

# 予測値を合成
par_pred1 = 0.65# pred1の割合
final_pred = pred1*par_pred1 + pred2*(1-par_pred1)

# 最初の予測値と合成後の予測値を見て変化の度合いを確認
plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.plot(range(1500),pred1, label="pred1")
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(1500),final_pred, label="final_pred [pred1="+str(par_pred1)+"]")
plt.legend()
plt.show()


# 提出

sample_df = pd.read_csv("path to sumple file",header=None)
sample_df[1]=pd.DataFrame(final_pred)

sample_df.to_csv("submit file name",index=False, header=False)