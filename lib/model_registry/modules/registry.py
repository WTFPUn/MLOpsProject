import mlflow
from mlflow.tracking import MlflowClient
from config import HF_REPO_ID_PARAM_NAME, EXPERIMENT_NAME

def get_current_model(model_name, metric, mode="MAXIMIZE"):
    client = MlflowClient()

    # Use alias '@production' to get the current production model
    alias = "Production"
    try:
        model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
    except Exception:
        raise ValueError(f"No model with alias '@{alias}' found for '{model_name}'")

    run = client.get_run(model_version.run_id)
    metric_value = run.data.metrics.get(metric)
    if metric_value is None:
        raise ValueError(f"Metric '{metric}' not found in production model run.")

    repo_id = run.data.params.get(HF_REPO_ID_PARAM_NAME)
    return repo_id, model_version.version, model_version

def get_runs_with_same_param_value(repo_name: str, exp_info):
    client = mlflow.tracking.MlflowClient()

    # Step 1: Get experiment by name
    experiment = client.get_experiment_by_name(exp_info.name)
    if not experiment:
        raise ValueError(f"Experiment '{exp_info.name}' not found")
    experiment_id = experiment.experiment_id

    # Search all runs with the same param value
    filter_str = f"params.{HF_REPO_ID_PARAM_NAME} = '{repo_name}'"
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_str
    )

    return runs_df

if __name__ == "__main__":
    print(EXPERIMENT_NAME)
    print(get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score", "MINIMIZE"))