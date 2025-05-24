
from datetime import datetime, timedelta
import os, subprocess, json, pickle, boto3

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from scraping.scraping import scrape_thairath
from news_cluster.embed_news import embed_colbert
from news_cluster.news_cluster import cluster_colbert_vectors
from model_registry.modules.registry import get_current_model

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
    dag_id              = "inference pipeline",
    start_date          = datetime(2024, 1, 1),
    schedule_interval   = "0 0 * * *",
    catchup             = False,
    default_args        = default_args,
    max_active_runs     = 1,
    tags                = ["news", "nlp", "topic-model"],
):
    
    @task()
    def load_data_from_s3():
      # s3 = boto3.client("s3")
      # try:
      #     s3.upload_file(csv_file, "kmuttcpe393datamodelnewssum",
      #      f"news_summary/{datetime.now().strftime('%d%m%Y')}.csv")
      # except Exception as e:
      #     print(f"❌ S3 upload failed: {e}")
      print("loading data...")
      return csv_file

    @task()
    def get_embedding_model(csv_file: str = ''):
        print("getting model from registry...")
        repo_name, _ , _= get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score", "MINIMIZE")
        return csv_file, repo_name
          
    @task()
    def topic_modelling(csv_file: str,repo_name: str):
        print("Topic modelling...")
        folder_name = embed_colbert(csv_file, repo_name)
        cluster_colbert_vectors(folder_name, "news_clustered.csv")

    @task()
    def summarise_topics():
      print("Summarising topics...")
      target_file = "news_clustered.csv"
      # get api end point
      # request
      summarized_text = None

      return summarized_text

    @task()
    def store_summaries(summarized_text: str):
        print("Storing summaries...")

    load_s3    = load_data_from_s3()
    model_registry = get_embedding_model(load_s3)
    topics_s3  = topic_modelling(model_registry)
    summary_s3 = summarise_topics()
    store_sum = store_summaries()
    
    load_s3 >> model_registry >> topics_s3 >> summary_s3 >> store_sum
