# scripts/inspect_data.py
import pandas as pd

def main():
    pub = pd.read_csv("big data/public_data.csv")
    priv = pd.read_csv("big data/private_data.csv")
    print("Public data shape:", pub.shape)
    print(pub.head(), "\n")
    print("Private data shape:", priv.shape)
    print(priv.head(), "\n")
    print("Public dtypes:\n", pub.dtypes, "\n")
    print("Private dtypes:\n", priv.dtypes, "\n")
    print("Public missing values:", pub.isna().sum().sum())
    print("Private missing values:", priv.isna().sum().sum())

if __name__ == "__main__":
    main()
