import argparse
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
import pickle
import os
import torch
import io

class Embeder:
    def __init__(self, model_name = 'BAAI/bge-m3'):
        self.model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BGEM3FlagModel(model_name, use_fp16=True, devices = device)

    def set_model(self, model_name):
        self.model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BGEM3FlagModel(model_name, use_fp16=True, devices = device)

    def embed_colbert(self, input_csv_path, model_name = 'BAAI/bge-m3', api=True):
        if api:
            df = input_csv_path.copy()
        else:
            print(f"📥 Reading: {input_csv_path}")
            df = pd.read_csv(input_csv_path)
        df = df.dropna(subset=["content"])
        texts = df["content"].tolist()

        if model_name != self.model_name:
            print(f"🧠 Loading model...")
            self.set_model(model_name)
        else:
            print(f"🧠 Cached model...")
        

        print(f"🔄 Embedding {len(texts)} texts using ColBERT vectors...")
        outputs = self.model.encode(texts, return_dense=False, return_sparse=False, return_colbert_vecs=True)
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
    parser.add_argument(default="input_csv", type=str, help="Path to the input CSV file (e.g., news_140525.csv)")

    args = parser.parse_args()
    test = Embeder()
    test.embed_colbert(args.input_csv, api = False)
