import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 讀取原始資料與聚類結果
df = pd.read_csv("data/public_data.csv")
labels = pd.read_csv("public_submission.csv")["label"]

# 前處理：標準化後做 PCA（不含 id 欄位）
Xz = StandardScaler().fit_transform(df.drop("id", axis=1))
pca2 = PCA(n_components=2).fit_transform(Xz)

# 繪圖：PCA 2D 投影圖，著色依據聚類標籤
plt.figure(figsize=(6, 5))
plt.scatter(
    pca2[:, 0], pca2[:, 1],
    c=labels,
    s=5,
    cmap="tab20",
    alpha=0.7,
    linewidths=0
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("圖Y：公共集 PCA2D 細分結果 (15 群)")
plt.colorbar(label="Cluster label", ticks=range(int(labels.max()) + 1))
plt.tight_layout()
plt.savefig("results/plots/pca2d_pub.png", dpi=150)
plt.show()
