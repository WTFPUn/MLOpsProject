"""
this program will run ml registry for news clustering with following psuedo code
on regular run
1. get best model from latest time stamp 
2. evaluate model on new dataset
3. evaluate performance of model on shifted data
4. log to mlflow
5. if model performance needed to be updated (performance greatly decrease)
    - trigger model finetuning (P'win)
    - run step 2-4 on new model
    - compare latest model score with current best
    - if better:
        - new model: production tag
        - current model: archieve tag
        - push new model to hub 

on initial run
1. detech if there is no experiment
2. run 2-4 as above
3. add production tag

Raises:
    ValueError: _description_

Returns:
    _type_: _description_
"""
import mlflow
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

HF_REPO_ID_PARAM_NAME = "huggingface_repo_id"
EXPERIMENT_NAME = "Vector_Prediction_Viz"
PLOT_FILENAME = "prediction_vectors.png"
PRODUCTION_TAG_KEY = "model_stage" # Common key for model stage
PRODUCTION_TAG_VALUE = "production"
REQUEST_TIMEOUT_SECONDS = 10 
os.environ["MLFLOW_CLIENT_REQUEST_TIMEOUT"] = str(REQUEST_TIMEOUT_SECONDS)
# TODO: add embedding plot
# def plot_embed(self, prediction):
#     print(f"Generating plot '{PLOT_FILENAME}'...")
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.scatter(prediction[:, 0], prediction[:, 1], alpha=0.7)
#     ax.set_title("2D Visualization of Prediction Vectors")
#     ax.set_xlabel("Dimension 1")
#     ax.set_ylabel("Dimension 2")
#     ax.grid(True)
#     plt.savefig(PLOT_FILENAME) # Save the plot to a file
#     plt.close(fig) # Close the plot to free memory
#     print("Plot saved.")

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
    # TODO: wait for scrape airflow directory and update data path/ data base query util function call ex. aws 
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

def get_current_model(
        EXPERIMENT_NAME,
        METRIC_TO_OPTIMIZE,
        OPTIMIZATION_MODE = "MAXIMIZE",
        ):
    # Connect to MLflow (assumes local `mlruns` or MLFLOW_TRACKING_URI is set)
    client = mlflow.tracking.MlflowClient()
    # TODO: add 
    # Get the Experiment ID
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")
        experiment_id = experiment.experiment_id
        print(f"Found Experiment ID: {experiment_id}")
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    # Construct the ordering string for the search query
    order_direction = "DESC" if OPTIMIZATION_MODE == "MAXIMIZE" else "ASC"
    order_by_string = f"metrics.`{METRIC_TO_OPTIMIZE}` {order_direction}" # Use backticks if metric name has special chars

    # Construct the filter string to find relevant runs
    filter_string = (
        f"tags.`{PRODUCTION_TAG_KEY}` = '{PRODUCTION_TAG_VALUE}' AND status = 'FINISHED'"
    )

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=[order_by_string],
        max_results=1,
    )

    # Extract information from the best run (which is the first row)
    best_run_row = runs_df.iloc[0]
    best_run_id = best_run_row['run_id']
    best_metric_value = best_run_row[f'metrics.{METRIC_TO_OPTIMIZE}']
    best_hf_repo_id = best_run_row[f'params.{HF_REPO_ID_PARAM_NAME}']

    print("\n--- Best Run Found ---")
    print(f"Run ID: {best_run_id}")
    print(f"Metric ({METRIC_TO_OPTIMIZE}): {best_metric_value}")
    print(f"Hugging Face Repo ID: {best_hf_repo_id}")
    print("--- Loading Model from Hugging Face Hub ---")
    if not best_hf_repo_id:
        best_hf_repo_id = "BAAI/bge-m3"
    # baseline = "BAAI/bge-m3"
    return best_hf_repo_id

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Starting MLflow run...")
    MODEL_REGISTRY_NAME = "BAAI/bge-m3"
    
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    init = False

    # set initial run flag
    if experiment is None:
        init = True
        print("running first experiment..")
        print(f"set first model as {MODEL_REGISTRY_NAME}")

        # Experiment does not exist, create it
        print(f"Experiment '{EXPERIMENT_NAME}' does not exist. Creating it.")
        try:
            experiment_id = client.create_experiment(EXPERIMENT_NAME)
            print(f"Experiment '{EXPERIMENT_NAME}' created with ID: {experiment_id}")
            is_first_ever_run_in_this_experiment = True # This run will be the first
            # Fetch the experiment object again now that it's created
            experiment = client.get_experiment(experiment_id)
        except mlflow.exceptions.MlflowException as e:
            # Handle race condition if another process created it just now
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                print(f"Experiment '{EXPERIMENT_NAME}' was created by another process. Fetching it.")
                experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
                if not experiment:
                    print(f"ERROR: Failed to create or get experiment '{EXPERIMENT_NAME}'. Exiting.")
                    exit()
                # Re-check runs for the now existing experiment (see below)
            else:
                raise e # Re-raise other MLflow exceptions
    else:
         # get current best mlflow registry name 
        MODEL_REGISTRY_NAME = get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score")
    
    NAME = experiment.name

    mlflow.set_experiment(EXPERIMENT_NAME)
    if is_challenged(): 
        with mlflow.start_run() as run:
            run_id = run.info.run_uuid
            print(f"MLflow Run ID: {run_id}")

            # load current model 
            model = Embedding_model(model_name=MODEL_REGISTRY_NAME)

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


            mlflow.log_param(HF_REPO_ID_PARAM_NAME, MODEL_REGISTRY_NAME)
            mlflow.set_tag("huggingface_repo_url", f"https://huggingface.co/{MODEL_REGISTRY_NAME}")

            # --- 6. Register the Model ---
            # This takes the logged model from the run and registers it
            model_uri = f"runs:/{run_id}/model"
            print(f"Registering model from URI: {model_uri}")
            registered_model_info = mlflow.register_model(
                model_uri=model_uri,
                name=NAME
            )
            print(f"Successfully registered model '{NAME}' Version: {registered_model_info.version}")
            # print(f"Model registered as '{MODEL_REGISTRY_NAME}' version {registered_model_info.version}")
            if init:
                mlflow.set_tag(PRODUCTION_TAG_KEY, PRODUCTION_TAG_VALUE)
                print(f"This is the first run in the experiment. Tagged with '{PRODUCTION_TAG_KEY}': '{PRODUCTION_TAG_VALUE}'.")

        print("MLflow logging complete.")

        # Cleanup the plot file
        if os.path.exists(PLOT_FILENAME):
            os.remove(PLOT_FILENAME)

if __name__ == "__main__":
    main()