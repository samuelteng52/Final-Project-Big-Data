# main.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_clustering(input_csv, out_csv, dim):
    # 读数据
    df = pd.read_csv(input_csv)
    # 假设第一列是 id，后面是特征
    ids = df['id']
    feats = df.columns.drop('id')

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats])

    # 聚类
    k = 4 * dim - 1
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # 保存成 id,label
    out = pd.DataFrame({'id': ids, 'label': labels})
    out.to_csv(out_csv, index=False)

if __name__ == "__main__":
    # 4 维公开数据 → 15 群
    run_clustering("data/public_data.csv", "r13922111_public.csv", dim=4)
    # 6 维私有数据 → 23 群
    run_clustering("data/private_data.csv", "r13922111_private.csv", dim=6)
    print("Done. Outputs written to r13922111_public.csv and r13922111_private.csv")
