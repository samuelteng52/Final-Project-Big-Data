import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans

def run(input_csv, output_csv, dim):
    df = pd.read_csv(input_csv)
    ids = df['id'].values
    X = df.drop('id', axis=1).values
    Xz = StandardScaler().fit_transform(X)
    
    k_coarse = 5
    agg2 = AgglomerativeClustering(n_clusters=k_coarse, linkage='ward')
    coarse_labels = agg2.fit_predict(Xz[:, [1,2]])
    
    final_labels = np.zeros_like(coarse_labels)
    for c in range(k_coarse):
        idx = np.where(coarse_labels == c)[0]
        X_sub = Xz[idx][:, [0,3]]  # S1, S4
        km = KMeans(n_clusters=3, random_state=42, n_init=500, max_iter=50,tol=1e-6)
        sub = km.fit_predict(X_sub)
        final_labels[idx] = c*3 + sub

    pd.DataFrame({'id': ids, 'label': final_labels}) \
      .to_csv(output_csv, index=False)
    print(f"Two-stage Agglo+KMeans done → {output_csv}")

if __name__ == "__main__":
    run("data/public_data.csv", "agglo_two_stage_public.csv", dim=4)
    #run("data/private_data.csv", "agglo_two_stage_private.csv", dim=6)
