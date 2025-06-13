import pandas as pd
import matplotlib.pyplot as plt

# 讀取原始資料與聚類結果
df = pd.read_csv("data/public_data.csv")
labels = pd.read_csv("public_submission.csv")["label"]

# 畫 S2 vs S3，並以聚類標籤著色
plt.figure(figsize=(6, 5))
plt.scatter(
    df["2"], df["3"],
    c=labels,
    s=5,
    cmap="tab20",
    alpha=0.7,
    linewidths=0
)
plt.xlabel("Feature 2 (S2)")
plt.ylabel("Feature 3 (S3)")
plt.title("公共資料集 S2 vs S3 聚類結果")
plt.colorbar(label="Cluster label", ticks=range(int(labels.max())+1))
plt.tight_layout()
plt.savefig("results/plots/23.png", dpi=150)
plt.show()
