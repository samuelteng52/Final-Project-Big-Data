# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

# def run_clustering(input_csv, out_csv, dim):
#     df = pd.read_csv(input_csv)
#     ids = df['id']
#     feats = df.columns.drop('id')

#     scaler = StandardScaler()
#     X = scaler.fit_transform(df[feats])

#     k = 4 * dim - 1
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = km.fit_predict(X)

#     # 这里确保 out 是 DataFrame，再写出 CSV
#     out = pd.DataFrame({"id": ids, "label": labels})
#     out.to_csv(out_csv, index=False)

# if __name__ == "__main__":
#     run_clustering("data/public_data.csv", "r13922111_public_1.csv", dim=4)
#     run_clustering("data/private_data.csv", "r13922111_private_1.csv", dim=6)
#     print("Done. Outputs written to r13922111_public.csv and r13922111_private.csv")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, Birch

def improved(input_csv, out_csv, dim):
    df = pd.read_csv(input_csv)
    ids = df['id'].values
    X = df.drop('id', axis=1).values

    # 1) 長尾壓縮 + 標準化
    X = np.log1p(X)
    Xz = StandardScaler().fit_transform(X)

    # 2) 降維 (PCA2)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xz)

    # 3) 加權 (可選)
    weights = np.array([1.0, 2.0, 2.0, 1.0] + ([1.0,1.0] if dim==6 else []))
    Xp = Xp * weights[:Xp.shape[1]]

    # 4) 聚類
    k = 4*dim - 1
    # 方案 A：MiniBatchKMeans
    mbk = MiniBatchKMeans(
        n_clusters=k, init="k-means++",
        random_state=42, n_init=20, batch_size=10000
    )
    labels = mbk.fit_predict(Xp)

    # 5) 輸出
    pd.DataFrame({'id': ids, 'label': labels}) \
      .to_csv(out_csv, index=False)

# 公共集
improved("data/public_data.csv", "improved_pub.csv", dim=4)
# 私有集
improved("data/private_data.csv", "improved_priv.csv", dim=6)
