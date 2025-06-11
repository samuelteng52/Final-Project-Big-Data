# scripts/eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt

def sample_and_plot(df, columns, sample_size=5000, prefix="pub"):
    # 隨機取 sample_size 筆作可視化，避免畫圖過久
    samp = df.sample(n=sample_size, random_state=42)
    for i, x in enumerate(columns):
        for y in columns[i+1:]:
            plt.figure(figsize=(6,4))
            plt.scatter(samp[x], samp[y], s=5)
            plt.xlabel(x); plt.ylabel(y)
            plt.title(f"{prefix} data: {x} vs {y}")
            out_path = f"results/plots/{prefix}_{x}_{y}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print("Saved", out_path)

def main():
    os.makedirs("results/plots", exist_ok=True)

    # 讀入公開資料
    pub = pd.read_csv("data/public_data.csv")
    # 假設前兩欄是 id + 4 維特徵，挑出特徵欄
    feat_pub = pub.columns.drop("id")  
    sample_and_plot(pub, feat_pub, sample_size=5000, prefix="pub")

    # 讀入私有資料（後面同理）
    priv = pd.read_csv("data/private_data.csv")
    feat_priv = priv.columns.drop("id")
    sample_and_plot(priv, feat_priv, sample_size=5000, prefix="priv")

if __name__ == "__main__":
    main()
