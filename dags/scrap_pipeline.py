
from datetime import datetime, timedelta
import os, boto3

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable

AWS_REGION         = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
GEMMA_MODEL_PATH   = ""

default_args = {
    "owner":           "SuperAIAinsurance",
    "depends_on_past": False,
    "retries":         2,
    "retry_delay":     timedelta(minutes=10),
}

with DAG(
  dag_id              = "scraping_pipeline",
  start_date          = datetime(2024, 1, 1),
  schedule_interval   = "0 0 * * 0",  # Run every Sunday at midnight
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

  @task(retries=1, retry_delay=timedelta(minutes=5))
  def scrape_news():
    from lib.scraping.scraping import scrape_thairath
    SCRAPING_DIR = os.path.join(os.path.dirname(__file__), "../scraping/data")
    os.makedirs(SCRAPING_DIR, exist_ok=True)

    current_date = datetime.now()
    out_path = os.path.join(
      SCRAPING_DIR,
      f"news_week_{date_parser(get_start_of_week(current_date))}.csv"
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
    current_date = datetime.now()
    try:
      s3.upload_file(csv_file, "kmuttcpe393datamodelnewssum",
       f"news_summary/news_week_{date_parser(get_start_of_week(current_date))}.csv")
    except Exception as e:
      print(f"❌ S3 upload failed: {e}")
      
      
  @task()
  def download_from_s3():
    s3 = boto3.client("s3")
    current_date = datetime.now()
    try:
      s3.download_file("kmuttcpe393datamodelnewssum",
       f"news_summary/news_week_{date_parser(get_start_of_week(current_date))}.csv",
       f"data/news_week_{date_parser(get_start_of_week(current_date))}.csv")
      print(f"✅ S3 download successful: news_week_{date_parser(get_start_of_week(current_date))}.csv")
    except Exception as e:
      print(f"❌ S3 download failed: {e}")

  raw_s3       = scrape_news()
  upload_s3    = upload_to_s3(raw_s3)
  
  raw_s3 >> upload_s3