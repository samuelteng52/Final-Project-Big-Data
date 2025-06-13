# scripts/cluster_pca2_kmeans.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run(input_csv, output_csv, dim):
    df = pd.read_csv(input_csv)
    ids = df['id']
    X = df.drop('id', axis=1).values
    # 1) 标准化
    X = StandardScaler().fit_transform(X)
    # 2) PCA 降到 2 维
    Xpca = PCA(n_components=2, random_state=42).fit_transform(X)
    # 3) KMeans (15 群)
    k = 4 * dim - 1
    km = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=500)
    labels = km.fit_predict(Xpca)
    # 4) 输出
    pd.DataFrame({'id': ids, 'label': labels}).to_csv(output_csv, index=False)
    print(f"PCA2+KMeans done → {output_csv}")

if __name__ == "__main__":
    run("data/public_data.csv", "pca2_kmeans_public.csv", dim=4)
    # 你也可以同时生成私有集：
    run("data/private_data.csv", "pca2_kmeans_private.csv", dim=6)
