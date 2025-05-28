from datetime import datetime, timedelta
import os, subprocess, json, pickle, boto3

from airflow.operators.python import PythonVirtualenvOperator
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable

import dotenv
dotenv_path = dotenv.find_dotenv()
if dotenv_path:
    print(f"Found .env file at {dotenv_path}")
    # export dotenv_path to environment variables
    dotenv.load_dotenv(dotenv_path)

EXPERIMENT_NAME    = "Vector_Prediction_Viz"
AWS_REGION         = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
GEMMA_MODEL_PATH   = ""
API_ENDPOINT         = os.getenv("API_ENDPOINT")

default_args = {
    "owner":           "SuperAIAinsurance",
    "depends_on_past": False,
    "retries":         2,
    "retry_delay":     timedelta(minutes=10),
}

with DAG(
    dag_id              = "model_monitor_pipeline",
    start_date          = datetime(2024, 1, 1),
    schedule_interval   = "30 0 * * 0",
    catchup             = False,
    default_args        = default_args,
    max_active_runs     = 1,
    tags                = ["news", "nlp", "topic-model"],
):    

    def load_data_from_s3():
        import boto3
        from datetime import datetime, timedelta
        from lib.scraping.scraping import date_parser, get_start_of_week
        
        print("Setting up S3 client and paths")
        s3 = boto3.client("s3")
        current_date = datetime.now() - timedelta(days=7)
        try:
          print(f"Downloading file from S3: news_week_{date_parser(get_start_of_week(current_date))}.csv")
          s3.download_file("kmuttcpe393datamodelnewssum",
          f"data/news_week_{date_parser(get_start_of_week(current_date))}.csv",
          f"data/news_week_{date_parser(get_start_of_week(current_date))}.csv")
          csv_file = f"data/news_week_{date_parser(get_start_of_week(current_date))}.csv"
          print(f"✅ S3 file downloaded to: {csv_file}")
        except Exception as e:
          print(f"❌ S3 download failed: {e}")
        import pandas as pd
        test = pd.read_csv(csv_file)
        print(test)
        return csv_file

    load_data_task = PythonVirtualenvOperator(
        task_id="load_test_monitor",
        python_callable=load_data_from_s3,
        requirements=["pandas"],
        python_version="3.10",
    )

    def test_model(csv_file: str):
        import sys
        sys.path.append("/opt/airflow/lib")
        sys.path.append("/opt/airflow/lib/model_registry")
        from lib.model_registry.model_trial import trial
        import os
        API_ENDPOINT = os.getenv("API_ENDPOINT")
        print(csv_file)
        trial(csv_file, API_ENDPOINT)
        # repo_name, _ , _= get_current_model("Vector_Prediction_Viz", "calinski_harabasz_score", "MINIMIZE")
        # print(f"found {repo_name} on production")
        # print(f"data is on : {csv_file}")
        # return {
        #     "csv_file": csv_file,
        #     "repo_name": repo_name
        # }

    get_model_task = PythonVirtualenvOperator(
        task_id="model_trial",
        python_callable=test_model,
        op_args=["{{ ti.xcom_pull(task_ids='load_test_monitor') }}"],
        requirements=["mlflow","pandas","numpy","scikit-learn==1.6.1"],
        python_version="3.10",
    )

    def compare_model():
        import sys
        sys.path.append("/opt/airflow/lib")
        sys.path.append("/opt/airflow/lib/model_registry")
        from lib.model_registry.monitor_model import check_embedding
        status = check_embedding()
        print(f"should we finetune?: {status}")
        # repo_name, _ , _= get_current_model("Vector_Prediction_Viz", "calinski_harabasz_score", "MINIMIZE")
        # print(f"found {repo_name} on production")
        # print(f"data is on : {csv_file}")
        # return {
        #     "csv_file": csv_file,
        #     "repo_name": repo_name
        # }

    compare_model_task = PythonVirtualenvOperator(
        task_id="model_comparison",
        python_callable=compare_model,
        # op_args=["{{ ti.xcom_pull(task_ids='load_test_monitor') }}"],
        requirements=["mlflow","pandas"],
        python_version="3.10",
    )

    load_data_task >> get_model_task >> compare_model_task
