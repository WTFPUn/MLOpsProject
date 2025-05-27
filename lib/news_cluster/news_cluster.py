import os
import pickle
from glob import glob
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import argparse


def cluster_colbert_vectors(input_folder, output_csv, as_folder = False):
    if as_folder:
        print(f"📂 Scanning folder: {input_folder}")
        pkl_files = sorted(glob(os.path.join(input_folder, "*.pkl")))

        if not pkl_files:
            print("❌ No PKL files found.")
            return

        # === Load and merge all PKL files ===
        print(f"🧩 Found {len(pkl_files)} PKL files. Loading...")
        all_df = []
        for file in pkl_files:
            with open(file, "rb") as f:
                all_df.append(pickle.load(f))
        merged_df = pd.concat(all_df, ignore_index=True)
    else:
        with open(input_folder, "rb") as f:
            merged_df = pickle.load(f)

    # Pool ColBERT vectors
    pooled_vecs = np.array([np.mean(vec, axis=0) for vec in merged_df["colbert_vecs"]])
    
    # Cluster with DBSCAN
    print("🔍 Clustering with DBSCAN...")
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
    labels = dbscan.fit_predict(pooled_vecs)

    merged_df["cluster"] = labels

    # Save result
    print(f"💾 Saving clustered results to {output_csv}")
    merged_df.to_csv(output_csv, index=False)
    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster ColBERT embeddings using mean pooling + UMAP + DBSCAN.")
    parser.add_argument("--input_folder", type=str, default="./embed_vector", help="Folder containing .pkl files (default: ./embed_vector)")
    parser.add_argument("--output_csv",  type=str, default="news_clustered.csv", help="Output CSV file (default: colbert_clustered.csv)")

    args = parser.parse_args()
    cluster_colbert_vectors(args.input_folder, args.output_csv)
    
