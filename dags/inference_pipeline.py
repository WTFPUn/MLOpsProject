
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

    @task()
    def get_embedding_model(csv_file: str = ''):
        from lib.model_registry.modules.registry import get_current_model
        print("getting model from registry...")
        repo_name, _ , _= get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score", "MINIMIZE")
        return {
            "csv_file": csv_file,
            "repo_name": repo_name
        }
          
    @task()
    def topic_modelling(input_data: dict):
        from lib.news_cluster.embed_news import embed_colbert
        from lib.news_cluster.news_cluster import cluster_colbert_vectors
        print("Topic modelling...")
        folder_name = embed_colbert(input_data["csv_file"], input_data["repo_name"])
        clustered_file = "news_clustered.csv"
        cluster_colbert_vectors(folder_name, clustered_file)
        return clustered_file

    @task()
    def summarise_topics(cluster_file: str):
      print(f"Summarising topics... on {cluster_file}")
      # get api end point
      # request
      summarized_text = cluster_file

      return summarized_text

    @task()
    def store_summaries(summarized_text: str):
        print("Storing summaries...")

    # Task chaining with proper passing
    csv_file = load_data_from_s3()
    model_info = get_embedding_model(csv_file)
    clustered_file = topic_modelling(model_info)
    summarized = summarise_topics(clustered_file)
    store_s3 = store_summaries(summarized)
    
    csv_file >> model_info >> clustered_file >> summarized >> store_s3
