# scripts/cluster.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_and_save(input_csv, output_csv, dim_count):
    # 读取数据（第一列 id，后面才是特征）
    df = pd.read_csv(input_csv)
    features = df.columns.drop("id")

    # 1) 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    # 2) K-Means 聚类
    k = 4 * dim_count - 1
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # 3) 保存结果
    pd.DataFrame({"cluster": labels}).to_csv(output_csv, index=False)
    print(f"{input_csv} → {output_csv} 完成，群数 = {k}")
    print(pd.Series(labels).value_counts().sort_index())

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # 公開資料：4 維 → 15 群
    cluster_and_save("data/public_data.csv", "public_submission.csv", dim_count=4)

    # 私有資料：6 維 → 23 群
    cluster_and_save("data/private_data.csv", "private_submission.csv", dim_count=6)
