import mlflow
from config import *
from modules.pipeline import run_experiment
from modules.registry import get_current_model
from modules.model import is_challenged

def main():
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
        model_name, _ = get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    if is_challenged():
        version, score = run_experiment(init, experiment.name, model_name)
        print(f"Model run complete: version={version}, score={score}")
        # Add comparison logic here

if __name__ == "__main__":

    main()
