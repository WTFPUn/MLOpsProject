
from datetime import datetime, timedelta
import os, subprocess, json, pickle, boto3

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
    schedule_interval   = "30 0 * * 0",
    catchup             = False,
    default_args        = default_args,
    max_active_runs     = 1,
    tags                = ["news", "nlp", "topic-model"],
):
    
    def get_start_of_week(date: datetime) -> datetime:
        start_of_week = date - timedelta(days=date.weekday())      
        return start_of_week

    def date_parser(dt: datetime) -> datetime:    
      return datetime.combine(dt.date(), datetime.min.time())
    
    @task()
    def load_data_from_s3():
        print("Setting up S3 client and paths")
        s3 = boto3.client("s3")
        current_date = datetime.now()
        try:
          print(f"Downloading file from S3: news_week_{date_parser(get_start_of_week(current_date))}.csv")
          s3.download_file("kmuttcpe393datamodelnewssum",
          f"news_summary/news_week_{date_parser(get_start_of_week(current_date))}.csv",
          f"data/news_week_{date_parser(get_start_of_week(current_date))}.csv")
          csv_file = f"data/news_week_{date_parser(get_start_of_week(current_date))}.csv"
          print(f"✅ S3 file downloaded to: {csv_file}")
        except Exception as e:
          print(f"❌ S3 download failed: {e}")
        return csv_file

    load_data_task = PythonVirtualenvOperator(
        task_id="load_data_from_s3",
        python_callable=load_data,
        requirements=["boto3"],
        python_version="3.10",
    )

    def get_model(csv_file: str):
        import sys
        sys.path.append("/opt/airflow/dag_lib")
        sys.path.append("/opt/airflow/dag_lib/model_registry")
        print("getting model from registry...")
        from model_registry.modules.registry import get_current_model
        repo_name, _ , _= get_current_model("Vector_Prediction_Viz", "calinski_harabasz_score", "MINIMIZE")
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

    # Task chaining with proper passing
    csv_file = load_data_from_s3()
    model_info = get_embedding_model(csv_file)
    clustered_file = topic_modelling(model_info)
    summarized = summarise_topics(clustered_file)
    store_s3 = store_summaries(summarized)
    
    csv_file >> model_info >> clustered_file >> summarized >> store_s3
