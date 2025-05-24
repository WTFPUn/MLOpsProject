from dotenv import load_dotenv
import os
import glob

# Find all .env files in the current directory
env_files = glob.glob("*.env")

if env_files:
    # Load the first .env file found
    print(f"Loading environment variables from: {env_files[0]}")
    load_dotenv(dotenv_path=env_files[0])
else:
    print("No .env file found.")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
PRODUCTION_TAG_KEY = os.getenv("PRODUCTION_TAG_KEY")
PRODUCTION_TAG_VALUE = os.getenv("PRODUCTION_TAG_VALUE")
HF_REPO_ID_PARAM_NAME = os.getenv("HF_REPO_ID_PARAM_NAME")
PLOT_FILENAME = os.getenv("PLOT_FILENAME")