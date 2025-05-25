
from datetime import datetime, timedelta
import os, subprocess, json, pickle, boto3
import sys
sys.path.append("/lib")

from airflow.operators.python import PythonVirtualenvOperator
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable

EXPERIMENT_NAME    = "Vector_Prediction_Viz"
AWS_REGION         = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
GEMMA_MODEL_PATH   = ""

default_args = {
    "owner":           "SuperAIAinsurance",
    "depends_on_past": False,
    "retries":         2,
    "retry_delay":     timedelta(minutes=10),
}

with DAG(
    dag_id              = "inference_pipeline",
    start_date          = datetime(2024, 1, 1),
    schedule_interval   = "0 0 * * *",
    catchup             = False,
    default_args        = default_args,
    max_active_runs     = 1,
    tags                = ["news", "nlp", "topic-model"],
):
    
    def load_data():
        print("loading data...")
        # TODO: implement this
        # get time range from request
        # query news dfrom S3 base on the time range
        # save to csv file

        # Simulate returning a CSV path
        return "/data/thairath_month.csv"

    load_data_task = PythonVirtualenvOperator(
        task_id="load_data_from_s3",
        python_callable=load_data,
        requirements=["boto3"],
        python_version="3.10",
    )

    def get_model(csv_file: str):
        import sys
        sys.path.append("/lib")
        print("getting model from registry...")
        from lib.model_registry.modules.registry import get_current_model
        repo_name, _ , _= get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score", "MINIMIZE")
        print(f"found {repo_name} on production")
        return {
            "csv_file": csv_file,
            "repo_name": repo_name
        }

    get_model_task = PythonVirtualenvOperator(
        task_id="get_embedding_model",
        python_callable=get_model,
        op_args=["{{ ti.xcom_pull(task_ids='load_data_from_s3') }}"],
        requirements=["mlflow"],
        python_version="3.10",
    )

    def topic_modeling(input_data: dict):
        print("Topic modelling...")
        clustered_file = "/tmp/news_clustered.csv"
        return clustered_file

    topic_model_task = PythonVirtualenvOperator(
        task_id="topic_modelling",
        python_callable=topic_modeling,
        op_args=["{{ ti.xcom_pull(task_ids='get_embedding_model') }}"],
        requirements=[],
        python_version="3.10",
    )

    def summarise(cluster_file: str):
        print(f"Summarising topics... on {cluster_file}")
        summarized_text = f"Summarized: {cluster_file}"
        return summarized_text

    summarise_task = PythonVirtualenvOperator(
        task_id="summarise_topics",
        python_callable=summarise,
        op_args=["{{ ti.xcom_pull(task_ids='topic_modelling') }}"],
        requirements=[],
        python_version="3.10",
    )

    def store_summary(text: str):
        print(f"Storing summary: {text}")

    store_summary_task = PythonVirtualenvOperator(
        task_id="store_summaries",
        python_callable=store_summary,
        op_args=["{{ ti.xcom_pull(task_ids='summarise_topics') }}"],
        requirements=[],
        python_version="3.10",
    )

    # Set task dependencies
    load_data_task >> get_model_task >> topic_model_task >> summarise_task >> store_summary_task
