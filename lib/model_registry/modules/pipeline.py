import mlflow
import os
from config import *
from modules.model import Embedding_model
from modules.dataset import get_test
from modules.evaluator import calculate_vector_spread
from dotenv import load_dotenv
import glob

# Find all .env files in the current directory
env_files = glob.glob("*.env")
if env_files:
    load_dotenv(dotenv_path=env_files[0])
else:
    print("No .env file found.")
DEBUG_PATH = os.getenv("DEBUG_PATH")

def run_experiment(init, name, model_name):
    client = mlflow.tracking.MlflowClient()    
    with mlflow.start_run() as run:
        mlflow.log_param(HF_REPO_ID_PARAM_NAME, model_name)
        mlflow.set_tag("huggingface_repo_url", f"https://huggingface.co/{model_name}")
        
        model = Embedding_model(model_name=model_name)
        test_data = get_test(debug_path=DEBUG_PATH)
        print("loaded test data...")
        embeddings = model.embed(test_data)
        clusters = model.predict(embeddings)

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
