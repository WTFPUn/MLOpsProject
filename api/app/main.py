from typing import Optional
from datetime import date
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response as FastAPIResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

import json

from models import News
from query import query_news, store_news

app = FastAPI()


# Instrumentator for automatic metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Custom Prometheus metrics with status_code label
REQUEST_COUNTER = Counter(
    "app_requests_total", "Total number of requests", ["endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint", "status_code"]
)

@app.get("/health")
async def health_check() -> FastAPIResponse:
    status_code = 200
    REQUEST_COUNTER.labels(endpoint="/health", status_code=str(status_code)).inc()
    with REQUEST_LATENCY.labels(endpoint="/health", status_code=str(status_code)).time():
        return FastAPIResponse(content='{"status": "ok"}', media_type="application/json", status_code=status_code)

@app.get("/news")
async def get_news(date: Optional[date] = None, test: Optional[bool] = False):
    news = await query_news(date, test)
    if news is None:
        status_code = 404
        REQUEST_COUNTER.labels(endpoint="/news", status_code=str(status_code)).inc()
        with REQUEST_LATENCY.labels(endpoint="/news", status_code=str(status_code)).time():
            return FastAPIResponse(content='{"error": "No news found"}', media_type="application/json", status_code=status_code)
    status_code = 200
    REQUEST_COUNTER.labels(endpoint="/news", status_code=str(status_code)).inc()
    with REQUEST_LATENCY.labels(endpoint="/news", status_code=str(status_code)).time():
        # return FastAPIResponse(content={"data": [i.model_dump_json() for i in news]}, media_type="application/json", status_code=status_code)
        return FastAPIResponse(content=json.dumps([i.model_dump_json() for i in news]), media_type="application/json", status_code=status_code)


@app.post("/news")
async def post_news(request: Request):
    data = await request.json()
    try:
        news = News(**data)
    except ValueError as e:
        status_code = 422
        REQUEST_COUNTER.labels(endpoint="/news", status_code=str(status_code)).inc()
        with REQUEST_LATENCY.labels(endpoint="/news", status_code=str(status_code)).time():
            return FastAPIResponse(content='{"error": "Invalid data"}', media_type="application/json", status_code=status_code)
    
    result = await store_news(title=news.title, content=news.content, date=news.date, cluster=news.cluster)
    if not result:
        status_code = 500
        REQUEST_COUNTER.labels(endpoint="/news", status_code=str(status_code)).inc()
        with REQUEST_LATENCY.labels(endpoint="/news", status_code=str(status_code)).time():
            return FastAPIResponse(content='{"error": "Failed to store news"}', media_type="application/json", status_code=status_code)
    status_code = 201
    REQUEST_COUNTER.labels(endpoint="/news", status_code=str(status_code)).inc()
    with REQUEST_LATENCY.labels(endpoint="/news", status_code=str(status_code)).time():
        return FastAPIResponse(content='{"status": "ok"}', media_type="application/json", status_code=status_code)


@app.get("/metrics")
def metrics():
    return FastAPIResponse(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
