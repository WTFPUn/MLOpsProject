import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Placeholder model
import os
import argparse
from model_manager import *
from utils.model import Embedding_model
from sklearn.metrics import calinski_harabasz_score
from pathlib import Path

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" # Or a local './mlruns' dir, or Databricks etc.
# To run locally without a server, comment the line above.
# To run with a server: mlflow server --host 127.0.0.1 --port 5000

MODEL_REGISTRY_NAME = "baseline"
EXPERIMENT_NAME = "Vector_Prediction_Viz"
MODEL_REGISTRY_NAME = "MyVectorPredictor"
PLOT_FILENAME = "prediction_vectors.png"

def plot_embed(self, prediction):
    print(f"Generating plot '{PLOT_FILENAME}'...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(prediction[:, 0], prediction[:, 1], alpha=0.7)
    ax.set_title("2D Visualization of Prediction Vectors")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True)
    plt.savefig(PLOT_FILENAME) # Save the plot to a file
    plt.close(fig) # Close the plot to free memory
    print("Plot saved.")

def calculate_vector_spread(vectors, label):
    """Custom metric: Average distance from the centroid."""
    score = calinski_harabasz_score(vectors, label)
    return score

def is_challenged():
    """this is called when we want to use conditional base maintenance"""
    # get current model
    # test with cUpdted testset
    # load past log (cool start initial lg 0 and register zeroshot as defuat model)
    # if performance drop at {percent} rate of previous result
    # return status of the need tfor model update
    return True

def get_test(debug=False):
    """make model prediction on past 1 week data

    Returns:
        list of string : 1 week data
    """
    sentences = []
    if debug:
        # ตอนนี้อ่าน Doc จาก txt ไปก่อน
        txt_files = list(Path("./testcase").glob("*.txt"))
        for txt_file in txt_files:
            with txt_file.open("r", encoding="utf-8") as f:
                full_text = f.read().strip()
                if full_text:
                    sentences.append(full_text)
        print(f"รวมได้ {len(sentences)} ไฟล์ข้อความ")

    return sentences

def get_current_model():
    baseline = "BAAI/bge-m3"
    return baseline

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Starting MLflow run...")
    mlflow.set_experiment(EXPERIMENT_NAME)

    if is_challenged():
        with mlflow.start_run() as run:
            run_id = run.info.run_uuid
            print(f"MLflow Run ID: {run_id}")
            # get current best mlflow registry name 
            MODEL_REGISTRY_NAME = get_current_model()

            # load current model 
            model = Embedding_model(model_name=MODEL_REGISTRY_NAME)

            # Log parameters (optional)
            # mlflow.log_param("num_vectors", n_vectors)

            # make model prediction on past 1 week data
            test_set = get_test(debug=True)

            test_set_embedding  = model.embed(test_set)

            cluster = model.predict(test_set_embedding)

            Ch = calculate_vector_spread(test_set_embedding, cluster)
            # print(Ch)
            # print(type(Ch))
            # Log custom metric
            mlflow.log_metric("calinski_harabasz_score", Ch)
            print(f"Logged metric: vector_spread={calinski_harabasz_score}")

            # Log the plot artifact
            mlflow.log_artifact(PLOT_FILENAME)
            print(f"Logged artifact: {PLOT_FILENAME}")

            # Log the model
            # signature = mlflow.models.infer_signature(X_train, model.predict(X_train)) # Optional
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model" # Name of the directory within the run's artifacts
                # signature=signature # Add if you want input/output schema
            )
            print("Logged model.")

            # --- 6. Register the Model ---
            # This takes the logged model from the run and registers it
            model_uri = f"runs:/{run_id}/model"
            print(f"Registering model from URI: {model_uri}")
            registered_model_info = mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_REGISTRY_NAME
            )
            print(f"Model registered as '{MODEL_REGISTRY_NAME}' version {registered_model_info.version}")

        print("MLflow logging complete.")

        # Cleanup the plot file
        if os.path.exists(PLOT_FILENAME):
            os.remove(PLOT_FILENAME)

if __name__ == "__main__":
    main()