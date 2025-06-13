import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv('data/public_data.csv')
X = StandardScaler().fit_transform(df.drop('id',axis=1))
X2 = PCA(2, random_state=42).fit_transform(X)

k=15
labels = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=500).fit_predict(X2)

# 3) 计算每个样本的 silhouette 分数
sil_vals = silhouette_samples(X2, labels)

# 4) 画出 silhouette plot
y_lower = 10
plt.figure(figsize=(6,5))
for i in range(k):
    ith_vals = sil_vals[labels==i]
    ith_vals.sort()
    y_upper = y_lower + ith_vals.shape[0]
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5*ith_vals.shape[0], str(i))
    y_lower = y_upper + 10
plt.axvline(sil_vals.mean(), color='red', linestyle='--')
plt.xlabel('Silhouette coefficient')
plt.ylabel('Cluster')
plt.title('Silhouette plot (PCA2+KMeans)')
plt.show()
