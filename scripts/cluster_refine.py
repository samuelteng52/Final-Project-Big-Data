# scripts/cluster_agglo.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import kneighbors_graph

def run(input_csv, output_csv, dim, use_connectivity=False):
    df = pd.read_csv(input_csv)
    ids = df['id'].values
    X = df.drop('id', axis=1).values

    # 标准化
    Xz = StandardScaler().fit_transform(X)

    # 第一阶段：Ward on S2vsS3
    k_coarse = 5
    if use_connectivity:
        # 构造稀疏 KNN 图，让 Agglo 只在局部合并
        # n_neighbors 可以调小以节省内存，或调大以保留更多结构
        connectivity = kneighbors_graph(Xz[:, [1,2]], n_neighbors=20, include_self=False)
        agg = AgglomerativeClustering(
            n_clusters=k_coarse,
            linkage='ward',
            connectivity=connectivity,
            compute_full_tree=True
        )
    else:
        agg = AgglomerativeClustering(n_clusters=k_coarse, linkage='ward')
    coarse_labels = agg.fit_predict(Xz[:, [1,2]])

    # 第二阶段：细分
    final_labels = np.zeros_like(coarse_labels)
    for c in range(k_coarse):
        idx = np.where(coarse_labels == c)[0]
        X_sub = Xz[idx]  # 可以改成全维 Xz[idx][:, [0,3]] 或全 4 维
        km = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=300)
        sub = km.fit_predict(X_sub)
        final_labels[idx] = c*3 + sub

    # 保存
    pd.DataFrame({'id': ids, 'label': final_labels}) \
      .to_csv(output_csv, index=False)
    print(f"Agglo+KMeans done → {output_csv}")

if __name__ == "__main__":
    # 公共集：不用 connectivity，数据量较小
    run("data/public_data.csv", "agglo_two_stage_public.csv", dim=4, use_connectivity=False)
    # 私有集：开启 connectivity 才能跑得动
    run("data/private_data.csv", "agglo_two_stage_private.csv", dim=6, use_connectivity=True)
