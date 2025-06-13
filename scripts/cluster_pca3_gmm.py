# scripts/cluster_pca3_gmm.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def run(input_csv, out_csv, dim):
    df = pd.read_csv(input_csv)
    ids = df['id']
    X = df.drop('id', axis=1).values
    X = StandardScaler().fit_transform(X)

    # 降到3维
    Xpca = PCA(n_components=3, random_state=42).fit_transform(X)

    # GMM 全协方差
    k = 4 * dim - 1
    gm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    labels = gm.fit_predict(Xpca)

    pd.DataFrame({'id': ids, 'label': labels}).to_csv(out_csv, index=False)
    print(f"PCA3+GMM done → {out_csv}")

if __name__ == "__main__":
    # 公开集：4维 → 15群
    run("data/public_data.csv", "pca3_gmm_public.csv", dim=4)
    # 私有集：6维 → 23群（同理）
    run("data/private_data.csv", "pca3_gmm_private.csv", dim=6)
