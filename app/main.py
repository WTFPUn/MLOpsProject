import gradio as gr
from botocore.exceptions import NoCredentialsError, ClientError
import boto3
import dotenv
import re
import pandas as pd
import os
import uuid

from pydantic import BaseModel
from typing import List
from datetime import datetime
import requests
import io

os.makedirs("csv", exist_ok=True)

class NewsRequest(BaseModel):
    id: int
    title: str
    content: str
    cluster: int
    startDate: datetime  # Use string for date in ISO format
    
class ListOfNewsRequest(BaseModel):
    data: List[NewsRequest]

dotenv_path = dotenv.find_dotenv()
if dotenv_path:
    print(f"Found .env file at {dotenv_path}")
    # export dotenv_path to environment variables
    dotenv.load_dotenv(dotenv_path)
    
prefix = "news_summary"
s3 = boto3.client('s3', region_name='ap-southeast-1')  # specify your region
bucket_name = 'kmuttcpe393datamodelnewssum'


def extract_date(filename):
    """
    Extracts the date from the filename using regex.
    
    Args:
        filename (str): The name of the file.
        
    Returns:
        str: The extracted date in 'YYYY-MM-DD' format, or None if not found.
    """
    print(f"Extracting date from filename: {filename}")
    match = re.search(r'\d{4}\d{2}\d{2}', filename)

    # get the date part from the match
    match = match.group(0) if match else None
    # chanhe the date format to DDMMYYYY to YYYY-MM-DD
    return match[4:]+"-"+match[2:4]+"-"+match[0:2] if match else None

def get_csv(cluster, date):
    print(f"Fetching CSV for cluster/{date}/{cluster}.csv")
    obj = s3.get_object(Bucket=bucket_name, Key=f"cluster/{date}/{cluster}.csv")
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    
    file_id = uuid.uuid4()  # Generate a unique file ID
    df.to_csv(f"csv/{file_id}.csv", index=False)  # Save to a temporary file
    
    return df, file_id
    
with gr.Blocks() as demo:
    state = gr.State([])
    data = gr.State({})
    file_id = gr.State(None)
    
    def query_groups(filter_value):
        url = "http://host.docker.internal:8000/getclusternames"  # Replace with your actual URL if different
        response = requests.get(url, params={"dt_str": filter_value})
        if response.status_code != 200:
            gr.Error(f"Error {response.status_code}: {response.json()}")
            return []
        out = ListOfNewsRequest(data=response.json()["data"]).data
        
        out_map = [(item.title, item.cluster) for item in out]
        data.value = {item.cluster: item for item in out}
        return out_map, out_map
    
    def show_group_info(group_name, dropdown):
        # print(f"Showing info for group: {group_name}")
        # print(data.value[group_name])
        
        df, file_id = get_csv(group_name, dropdown)
        if df.empty:
            return "No data available for this group.", gr.DataFrame(visible=False), None
        
        return data.value[group_name].content if group_name in data.value else "No content available", df, gr.update(value=f"csv/{file_id}.csv", interactive=True)
        # return group_name
    
    def load_date():
        try:
            response = requests.get("http://host.docker.internal:8000/getdatelist")  # Replace with your actual URL if different
            if response.status_code == 200:
                dates = response.json()["data"]
                if not dates:
                    return []
                return dates           
            else:
                return []
        except (NoCredentialsError, ClientError) as e:
            return []

    with gr.Column():
        
        with gr.Column(scale=2):
            dropdown = gr.Dropdown(choices=load_date(), label="Select Date", interactive=True, value=None)
        with gr.Row(scale=1):
            query_btn = gr.Button("Query", variant="primary")
            
    with gr.Row(max_height=500):
        with gr.Column(scale=1):
            gr.Markdown("### Group Cards")
            group_cards = gr.Radio(choices=[], label="Groups", interactive=True)

        with gr.Column(scale=2):
            with gr.Blocks():
                gr.Markdown("### Group Info")
                group_info = gr.Markdown("No data to show now...", elem_id="group_info")
                table = gr.DataFrame(
                    headers=["title", "url", "scraped_at", "content", "tags"],
                    visible=True
                )
                download = gr.DownloadButton("Export CSV", value=None, interactive=True, variant="primary")
    def update_group_cards(filter_value):
        choice, data = query_groups(filter_value)
        return gr.update(choices=choice, value=None), data, gr.update(value=""), gr.DataFrame(visible=False), gr.update(value=None, interactive=False)

    query_btn.click(fn=update_group_cards, inputs=dropdown, outputs=[group_cards, data, group_info])
    group_cards.change(fn=show_group_info, inputs=[group_cards, dropdown], outputs=[group_info, table, download])

    
demo.launch(server_name="0.0.0.0")
