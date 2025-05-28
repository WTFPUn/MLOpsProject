import mlflow
import os
import sys
from config import *
# from modules.model import Embedding_model
Embedding_model = None
from modules.dataset import get_test
from modules.evaluator import calculate_vector_spread
import pandas as pd
import io
import requests
import pickle
import numpy as np

def run_experiment(init, name, model_name, csv_file, API_EP = None):
    client = mlflow.tracking.MlflowClient()    
    with mlflow.start_run() as run:
        mlflow.log_param(HF_REPO_ID_PARAM_NAME, model_name)
        mlflow.set_tag("huggingface_repo_url", f"https://huggingface.co/{model_name}")
        test_data = get_test(csv_file)
        if not API_EP:
            model = Embedding_model(model_name=model_name)
            print("loaded test data...")
            embeddings = model.embed(test_data)
            clusters = model.predict(embeddings)
        else:
            df = pd.DataFrame({"content":test_data})
            # ✅ Step 2: Convert DataFrame to CSV bytes
            csv_bytes = io.BytesIO()
            df.to_csv(csv_bytes, index=False)
            csv_bytes.seek(0)
            print("begin sending request")

            # ✅ Step 3: Send to FastAPI using multipart/form-data
            response = requests.post(
                f"{API_EP}/embed/",
                files={"file": ("input.csv", csv_bytes, "text/csv")},
                data={"repo_name": model_name}  
            )

            # ✅ Step 4: Handle response
            with open("embedding_output.csv", "wb") as f:
                f.write(response.content)
            
            print("received embedding from server...")
            with open("embedding_output.csv", "rb") as f:
                merged_df = pickle.load(f)
            embeddings = np.array([np.mean(vec, axis=0) for vec in merged_df["colbert_vecs"]])

            from lib.news_cluster.news_cluster import cluster_colbert_vectors
            cluster_colbert_vectors("embedding_output.csv", "output.csv")
            print("done clustering...")

            out = pd.read_csv("output.csv")
            clusters = out["cluster"].tolist()

        score = calculate_vector_spread(embeddings, clusters)
        mlflow.log_metric("calinski_harabasz_score", score)

        model_uri = f"runs:/{run.info.run_uuid}/model"
        model_info = mlflow.register_model(model_uri=model_uri, name=name)

        if init:
            client.set_registered_model_alias(
                name=EXPERIMENT_NAME,
                alias="Production",
                version="1"
            )
            print("Transition successful!")

            # mlflow.set_tag(PRODUCTION_TAG_KEY, PRODUCTION_TAG_VALUE)

        return model_info.version, score
