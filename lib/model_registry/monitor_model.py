from config import *
from modules.registry import get_current_model,  get_runs_with_same_param_value
import mlflow

def check_embedding(debug = False):
    """use to check if the embedding model needed to be finetuned

    Returns:
        Boolean: the new model status (OK (True)/Worse (False))
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    model_name, current_log_version, exp_info = get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score")
    runs = get_runs_with_same_param_value(model_name, exp_info)
    runs = runs.dropna(subset="metrics.calinski_harabasz_score")

    # if this is first version terminate
    if len(runs)<2:
        print("only 1 model existed")
        exit()

    print("checking score of both runs")
    latest_records = runs.sort_values(by="end_time", ascending=False).head(2)
    print(latest_records)
    score = latest_records["metrics.calinski_harabasz_score"].values
    current_score = score[0]
    # Higher CH score → better cluster separation (good embedding). diff (+)
    # Lower CH score → more overlapping or compact clusters (bad embedding or potential drift). diff (-)
    diff = current_score - score[1]
    print(f"found different: {diff}")
    # the performance decrease
    if diff < 0:
        if debug:
            print("model score decrease")
        std = runs["metrics.calinski_harabasz_score"].std()
        mean = runs["metrics.calinski_harabasz_score"].mean()
        threshold = mean - 2*std
        # on 95% confidence that the calinski_harabasz_score has decreased
        if current_score < threshold:
            if debug:
                print("the model performance is decrease with 95% confident")
            # trigger finetune airflow
            return True
        else:
            return False
    else:
        return False
        
if __name__ == "__main__":
    print(check_embedding())


    # # 1. Get the Model Version object
    # model_version_details = client.get_model_version(
    #     name=exp_name,
    #     version=MODEL_VERSION
    # )

    # if not model_version_details:
    #     print(f"Model Version '{MODEL_VERSION}' for model '{model_name}' not found.")
    #     exit()

    # print(f"\n--- Model Version Details ---")
    # print(f"Registered Model Name: {model_version_details.name}")
    # print(f"Version: {model_version_details.version}")
    # print(f"Status: {model_version_details.status}")
    # print(f"Current Stage: {model_version_details.current_stage}")
    # print(f"Creation Timestamp: {model_version_details.creation_timestamp}")
    # print(f"Source Run URI (from source attribute): {model_version_details.source}")

    # # 2. Extract the run_id
    # # The model_version_details object should have a run_id attribute directly
    # source_run_id = model_version_details.run_id
    # if not source_run_id:
    #     # Fallback: Try to parse from the 'source' URI if run_id is not directly available
    #     # (Usually, run_id is directly available)
    #     if model_version_details.source and model_version_details.source.startswith("runs:/"):
    #         source_run_id = model_version_details.source.split('/')[1]
    #     else:
    #         print("Could not determine the source run ID for this model version.")
    #         exit()

    #     print(f"Source Run ID: {source_run_id}")


    # # 3. Get the Run object
    # run_details = client.get_run(source_run_id)

    # if not run_details:
    #     print(f"Run with ID '{source_run_id}' not found.")
    #     exit()

    # print(f"\n--- Source Run Details ---")
    # print(f"Run ID: {run_details.info.run_id}")
    # print(f"Run Name: {run_details.data.tags.get('mlflow.runName', 'N/A')}") # Run name is a tag
    # print(f"Run Status: {run_details.info.status}")
    # print(f"Run Start Time: {run_details.info.start_time}")
    # metrics = run_details.data.metrics
    # print(f"Score: {metrics}")
    # prev_score = metrics["calinski_harabasz_score"]
    # # 4. Get the Experiment ID and details
    # source_experiment_id = run_details.info.experiment_id
    # experiment_details = client.get_experiment(source_experiment_id)

    # if not experiment_details:
    #     print(f"Experiment with ID '{source_experiment_id}' not found.")
    #     exit()

    # print(f"\n--- Source Experiment Details ---")
    # print(f"Experiment ID: {experiment_details.experiment_id}")
    # print(f"Experiment Name: {experiment_details.name}")
    # print(f"Experiment Artifact Location: {experiment_details.artifact_location}")