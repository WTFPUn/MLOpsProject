
from datetime import datetime, timedelta
import os, subprocess, json, pickle, boto3

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from scraping.scraping import scrape_thairath

AWS_REGION         = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
GEMMA_MODEL_PATH   = ""

default_args = {
    "owner":           "SuperAIAinsurance",
    "depends_on_past": False,
    "retries":         2,
    "retry_delay":     timedelta(minutes=10),
}

with DAG(
    dag_id              = "pipeline",
    start_date          = datetime(2024, 1, 1),
    schedule_interval   = "0 0 * * *",
    catchup             = False,
    default_args        = default_args,
    max_active_runs     = 1,
    tags                = ["news", "nlp", "topic-model"],
):

    @task(retries=1, retry_delay=timedelta(minutes=5))
    def scrape_news():
      SCRAPING_DIR = os.path.join(os.path.dirname(__file__), "../scraping/data")
      os.makedirs(SCRAPING_DIR, exist_ok=True)

      out_path = os.path.join(
          SCRAPING_DIR,
          f"news_{datetime.now().strftime('%d%m%y')}.csv"
      )
      try:   
        scrape_thairath(outpath=out_path)
        return out_path
      except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return out_path
    
    @task()
    def upload_to_s3(csv_file: str = ''):
      s3 = boto3.client("s3")
      try:
          s3.upload_file(csv_file, "kmuttcpe393datamodelnewssum",
           f"news_summary/{datetime.now().strftime('%d%m%Y')}.csv")
      except Exception as e:
          print(f"❌ S3 upload failed: {e}")
          
    @task()
    def bert_annotate():
        print("Annotation...")
        
    @task()
    def topic_modelling():
      print("Tropic modelling...")

    @task()
    def summarise_topics():
      print("Summarising topics...")

    @task()
    def store_summaries():
        print("Storing summaries...")

    raw_s3       = scrape_news()
    upload_s3    = upload_to_s3(raw_s3)
    bert_s3      = bert_annotate()
    topics_s3    = topic_modelling()
    summary_s3   = summarise_topics()
    store_sum = store_summaries()
    
    raw_s3 >> upload_s3 >> bert_s3 >> topics_s3 >> summary_s3 >> store_sum
