import mlflow
from config import *
from modules.pipeline import run_experiment
from modules.registry import get_current_model
# from modules.model import is_challenged
import sys
sys.path.append("./lib/model_registry")


def trial(csv_file, API_EP = None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    init = False

    if not experiment:
        print("Creating new experiment...")
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        experiment = client.get_experiment(experiment_id)
        model_name = "BAAI/bge-m3"
        init = True
    else:
        model_name, model_version , exp_info = get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"begin experimenting on model {model_name}")
    # if is_challenged():
    version, score = run_experiment(init, experiment.name, model_name, csv_file, API_EP)
    print(f"Model run complete: version={version}, score={score}")
    # Add comparison logic here
    return version, score

if __name__ == "__main__":
    trial()
