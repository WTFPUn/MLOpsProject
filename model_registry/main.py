import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Placeholder model
import os
import argparse
from model_manager import *
# --- Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" # Or a local './mlruns' dir, or Databricks etc.
# To run locally without a server, comment the line above.
# To run with a server: mlflow server --host 127.0.0.1 --port 5000


EXPERIMENT_NAME = "Vector_Prediction_Viz"
MODEL_REGISTRY_NAME = "MyVectorPredictor"
PLOT_FILENAME = "prediction_vectors.png"

# --- 3. Define and Calculate Custom Metric ---
def calculate_vector_spread(vectors):
    """Custom metric: Average distance from the centroid."""
    if vectors.shape[0] < 2:
        return 0.0
    centroid = np.mean(vectors, axis=0)
    distances = np.linalg.norm(vectors - centroid, axis=1)
    return np.mean(distances)

# --- 4. Generate 2D Plot ---
def plot_embed(prediction):
    print(f"Generating plot '{PLOT_FILENAME}'...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predictions[:, 0], predictions[:, 1], alpha=0.7)
    ax.set_title("2D Visualization of Prediction Vectors")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True)
    plt.savefig(PLOT_FILENAME) # Save the plot to a file
    plt.close(fig) # Close the plot to free memory
    print("Plot saved.")

def is_challenged():
    # get current model
    # test with cUpdted testset
    # load past log (cool start initial lg 0 and register zeroshot as defuat model)
    # if performance drop at {percent} rate of previous result
    # return status of the need tfor model update
    pass 

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # --- 5. Log with MLflow ---
    print("Starting MLflow run...")
    mlflow.set_experiment(EXPERIMENT_NAME)
    if is_challenged():

        with mlflow.start_run() as run:
            run_id = run.info.run_uuid
            print(f"MLflow Run ID: {run_id}")

            # Log parameters (optional)
            mlflow.log_param("num_vectors", n_vectors)

            # Log custom metric
            mlflow.log_metric("vector_spread", custom_metric_value)
            print(f"Logged metric: vector_spread={custom_metric_value:.4f}")

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