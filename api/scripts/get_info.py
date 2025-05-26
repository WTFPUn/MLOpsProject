import requests
from pydantic import BaseModel
from typing import List
from datetime import datetime
import json

class NewsRequest(BaseModel):
    id: int
    title: str
    content: str
    cluster: int
    startDate: datetime  # Use string for date in ISO format
    
class ListOfNewsRequest(BaseModel):
    data: List[NewsRequest]

# Define the URL of your FastAPI endpoint
url = "http://127.0.0.1:8000/getclusternames"  # Replace with your actual URL if different

# Example date string
dt_str = "2025-05-19"

# Send a GET request to the FastAPI endpoint with the date string as a query parameter
response = requests.get(url, params={"dt_str": dt_str})

# Check the response status code
if response.status_code == 200:
    # print("Response data:", ListOfNewsRequest(data=response.json()))
    print("Response data:", ListOfNewsRequest(data=response.json()["data"]))
    # print(json.dumps(out[0])) 
    # print("Response data:", len(out))  
else:
    print(f"Error {response.status_code}: {response.json()}")