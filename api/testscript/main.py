import requests
from datetime import datetime


news_data = {
    "title": "Sample News Title",
    "content": "This is a sample news description.",
    "date": datetime.now().isoformat()  # Use ISO format for date
}

response = requests.post("http://localhost:8000/news", json=news_data)
print(f"Status code: {response.status_code}")
print("Response body:", response.text)
