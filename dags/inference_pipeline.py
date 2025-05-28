
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
    dag_id              = "inference_pipeline",
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
        current_date = datetime.now()
        try:
          print(f"Downloading file from S3: news_week_{date_parser(get_start_of_week(current_date))}.csv")
          s3.download_file(
            "kmuttcpe393datamodelnewssum",
            f"data/news_week_{date_parser(get_start_of_week(current_date))}.csv",
            f"data/news_week_{date_parser(get_start_of_week(current_date))}.csv"
          )
          
          csv_file = f"data/news_week_{date_parser(get_start_of_week(current_date))}.csv"
          print(f"✅ S3 file downloaded to: {csv_file}")
        except Exception as e:
          print(f"❌ S3 download failed: {e}")
        return csv_file

    load_data_task = PythonVirtualenvOperator(
        task_id="load_data_s3",
        python_callable=load_data_from_s3,
        requirements=["boto3"],
        python_version="3.10",
    )

    def get_model(csv_file: str):
        import sys
        sys.path.append("/opt/airflow/lib")
        sys.path.append("/opt/airflow/lib/model_registry")
        print("getting model from registry...")
        from model_registry.modules.registry import get_current_model
        repo_name, _ , _= get_current_model("Vector_Prediction_Viz", "calinski_harabasz_score", "MINIMIZE")
        print(f"found {repo_name} on production")
        print(f"data is on : {csv_file}")
        return {
            "csv_file": csv_file,
            "repo_name": repo_name
        }

    get_model_task = PythonVirtualenvOperator(
        task_id="get_embedding_model",
        python_callable=get_model,
        op_args=["{{ ti.xcom_pull(task_ids='load_data_s3') }}"],
        requirements=["mlflow"],
        python_version="3.10",
    )

    def get_topic_embedding(input_data: dict):
        import os
        API_ENDPOINT = os.getenv("API_ENDPOINT")
        import ast
        input_data = ast.literal_eval(input_data)
        print("Topic modelling...")
        clustered_file = input_data["csv_file"]#"/tmp/news_clustered.csv"
        repo_name = input_data["repo_name"]
        import io
        import pandas as pd
        import requests
        print(clustered_file)

        # ✅ Step 1: Create or have a DataFrame
        df = pd.read_csv(clustered_file).sample(20)

        print(df)

        # ✅ Step 2: Convert DataFrame to CSV bytes
        csv_bytes = io.BytesIO()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)

        # # ✅ Step 3: Send to FastAPI using multipart/form-data
        response = requests.post(
            f"{API_ENDPOINT}/embed/",
            files={"file": ("input.csv", csv_bytes, "text/csv")},
            data={"repo_name": repo_name},
        )

        # import base64

        # res_json = response.json()
        # with open("embedding_output.pkl", "wb") as f:
        #     f.write(base64.b64decode(res_json["content"]))

        # # ✅ Step 4: Handle response
        with open("embedding_output.csv", "wb") as f:
            f.write(response.content)
        print("Processed CSV saved.")

        # from lib.news_cluster.embed_news import embed_colbert
        # embedding = embed_colbert(clustered_file, repo_name)
        # print(embedding)
        # from lib.news_cluster.embed_news 
        return "embedding_output.csv"

    embed_task = PythonVirtualenvOperator(
        task_id="get_topic_embedding",
        python_callable=get_topic_embedding,
        op_args=["{{ ti.xcom_pull(task_ids='get_embedding_model') }}"],
        requirements=["pandas"],
        python_version="3.10",
    )

    def get_cluster(file_path: str):
        from lib.news_cluster.news_cluster import cluster_colbert_vectors
        cluster_colbert_vectors(file_path, "output.csv")
        return "output.csv"

    topic_cluster_task = PythonVirtualenvOperator(
        task_id="topic_modelling",
        python_callable=get_cluster,
        op_args=["{{ ti.xcom_pull(task_ids='get_topic_embedding') }}"],
        requirements=["argparse","numpy","pandas","scikit-learn==1.6.1"],
        python_version="3.10",
    )

    def summarise(cluster_file: str):
        from typing import TypedDict, List
        import uuid
        import json

        class SummaryRecord(TypedDict):
            title: str
            cluster_id: int
            summarized_news: str  # Full LLM output for the cluster
            date: str

            
        print(f"Summarising topics... on {cluster_file}")
        summarized_text = "summarized_text.csv"
        import os
        import requests

        API_ENDPOINT = os.getenv("API_ENDPOINT")
        DEMO_N_CLUSTER = os.getenv("DEMO_N_CLUSTER")

        import io
        import pandas as pd

        # ✅ Step 1: Create or have a DataFrame
        df = pd.read_csv(cluster_file)

        # loop sent cluster
        summarized_texts =[]
        unique_cluster_ids = sorted(df['cluster'].unique())
        l = min(len(unique_cluster_ids),int(DEMO_N_CLUSTER))
        to_return = []
        csv_return = []
        print(f"Summarising {l} topics...")
        for cluster_id_val in unique_cluster_ids[:l+1]:
            if cluster_id_val == -1:
                print("skip individual cluster")
                continue
            target_df = df[df["cluster"] == cluster_id_val]

            # ✅ Step 2: Convert DataFrame to CSV bytes
            csv_bytes = io.BytesIO()
            target_df.to_csv(csv_bytes, index=False)
            csv_bytes.seek(0)
            
            csv_return.append(target_df)

            # ✅ Step 3: Send to FastAPI using multipart/form-data
            response = requests.post(
                f"{API_ENDPOINT}/summarize-news/",
                files={"file": ("input.csv", csv_bytes, "text/csv")},
                data={"output_path": summarized_text}  
            )

            # Check for response status
            if response.status_code != 200:
                print("Request failed:", response.status_code)
                print("Response text:", response.text)
                raise Exception("Request to summarize-news failed.")

            # response_df = pd.read_csv(io.BytesIO(response.content))
            # summarized_texts.append(response_df)
            if len(response.json()["data"]) == 0:
                print(f"No data returned for cluster {cluster_id_val}, skipping...")
                continue
            response: List[SummaryRecord] = response.json()["data"][0]
            print(f"Cluster {cluster_id_val} has {len(response)} records")           
            to_return.append(response)
            print(f"successfully summarized cluster {cluster_id_val}")

        # Save the returned CSV
        # all_df = pd.concat(summarized_texts)
        # all_df.to_csv(summarized_text)
        # with open(summarized_text, "wb") as f:
        #     f.write(response.content)
        # with open(cluster_file, "rb") as f:
        #     response = requests.post(
        #         f"{API_ENDPOINT}/summarize-news/",
        #         files={"file": ("input.pkl", f, "application/octet-stream")},
        #         data={"output_path": summarized_text}
        #     )
        print(to_return)
        
        to_save = {
            "df": df.to_dict(orient="records"),
            "to_return": to_return,  # already list of dicts from API
            "csv_return": [d.to_dict(orient="records") for d in csv_return]
        }
        serialized_data = json.dumps(to_save)
        file_name = f"summarized_{uuid.uuid4()}.json"
        with open(file_name, "w") as f:
            f.write(serialized_data)
        
        return file_name
        # return {
        #     "df": df,
        #     "to_return": to_return,
        #     "csv_return": csv_return
        # }

    summarise_task = PythonVirtualenvOperator(
        task_id="summarise_topics",
        python_callable=summarise,
        op_args=["{{ ti.xcom_pull(task_ids='topic_modelling') }}"],
        requirements=["pandas", "requests"],
        python_version="3.10",
    )

    def upload_to_s3(filename: str):
        import boto3
        from datetime import datetime
        import io
        from lib.scraping.scraping import date_parser, get_start_of_week
        import json
        import pandas as pd
        import ast
        
        # input_data = ast.literal_eval(input_data)
        # print(input_data)
        # df = input_data["df"]
        # to_return = input_data["to_return"]
        # csv_return = input_data["csv_return"]
        with open(filename, "r") as f:
            input_data = json.load(f)
        
        df = input_data["df"]
        to_return = input_data["to_return"]
        csv_return = input_data["csv_return"]
    
        df = pd.DataFrame(df)
        csv_return = [pd.DataFrame(x) for x in csv_return]
    
        print("to_return: ", to_return)
        print("csv_return: ", csv_return)
        
        s3 = boto3.client("s3")
        current_date = datetime.now()
        print(current_date)
        try:
            csv_bytes = io.BytesIO()
            df.to_csv(csv_bytes, index=False)
            s3.upload_file(csv_bytes, "kmuttcpe393datamodelnewssum",
            f"news_summary/news_week_{date_parser(get_start_of_week(current_date))}.csv"
            )
            
        except Exception as e:
            print(f"❌ S3 upload failed: {e}")
        
    store_summary_task = PythonVirtualenvOperator(
        task_id="store_summaries",
        python_callable=upload_to_s3,
        op_args=["{{ ti.xcom_pull(task_ids='summarise_topics') }}"],
        requirements=[],
        python_version="3.10",
    )

    # Task chaining with proper passing
    # csv_file = load_data_from_s3()
    # model_info = get_embedding_model(csv_file)
    # clustered_file = topic_modelling(model_info)
    # summarized = summarise_topics(clustered_file)
    # store_s3 = store_summaries(summarized)
    
    load_data_task >> get_model_task >> embed_task >> topic_cluster_task >> summarise_task >> store_summary_task#>> clustered_file >> summarized >> store_s3
