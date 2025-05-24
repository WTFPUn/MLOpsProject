import argparse
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
import pickle
import os


def embed_colbert(input_csv_path, model_name = 'BAAI/bge-m3'):
    print(f"📥 Reading: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    df = df.dropna(subset=["content"])
    texts = df["content"].tolist()

    print(f"🧠 Loading model...")
    model = BGEM3FlagModel(model_name, use_fp16=True)

    print(f"🔄 Embedding {len(texts)} texts using ColBERT vectors...")
    outputs = model.encode(texts, return_dense=False, return_sparse=False, return_colbert_vecs=True)
    colbert_vecs = outputs["colbert_vecs"]

    df["colbert_vecs"] = colbert_vecs

    # Create output filename
    base_name = os.path.basename(input_csv_path).replace(".csv", "")
    os.makedirs("embed_vector", exist_ok=True)
    output_pkl_path = f"./embed_vector/{base_name}.pkl"

    print(f"💾 Saving to: {output_pkl_path}")
    with open(output_pkl_path, "wb") as f:
        pickle.dump(df, f)

    print("✅ Done!")

    return output_pkl_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed daily news using ColBERT vectors.")
    parser.add_argument("input_csv", help="Path to the input CSV file (e.g., news_140525.csv)")

    args = parser.parse_args()
    embed_colbert(args.input_csv)
