import mlflow
from config import PRODUCTION_TAG_KEY, PRODUCTION_TAG_VALUE, HF_REPO_ID_PARAM_NAME, EXPERIMENT_NAME

def get_current_model(experiment_name, metric, mode="MAXIMIZE"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")
    
    order = "DESC" if mode == "MAXIMIZE" else "ASC"
    filter_str = f"tags.`{PRODUCTION_TAG_KEY}` = '{PRODUCTION_TAG_VALUE}' AND status = 'FINISHED'"
    order_str = f"metrics.`{metric}` {order}"

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=[order_str],
        max_results=1,
    )

    if runs_df.empty:
        raise ValueError("No production runs found.")

    best = runs_df.iloc[0]
    model_uri = f"runs:/{best.run_id}/model"
    model_info = mlflow.register_model(model_uri=model_uri, name=experiment_name)

    return best[f'params.{HF_REPO_ID_PARAM_NAME}'], model_info.version

if __name__ == "__main__":
    print(get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score", "MINIMIZE"))
