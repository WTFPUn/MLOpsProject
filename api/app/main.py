from typing import Optional
from datetime import date
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response as FastAPIResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from datetime import datetime
import json

from models import News, GetClusterNames
from query import query_news, store_news, get_news_by_date, query_date_list

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

@app.get("/getdatelist")
async def get_date_list():
    """
    Get a list of dates for which news is available.
    """
    try:
        dates = await query_date_list()
        if not dates:
            status_code = 404
            REQUEST_COUNTER.labels(endpoint="/getdatelist", status_code=str(status_code)).inc()
            with REQUEST_LATENCY.labels(endpoint="/getdatelist", status_code=str(status_code)).time():
                return FastAPIResponse(content='{"error": "No dates found"}', media_type="application/json", status_code=status_code)
        
        status_code = 200
        REQUEST_COUNTER.labels(endpoint="/getdatelist", status_code=str(status_code)).inc()
        with REQUEST_LATENCY.labels(endpoint="/getdatelist", status_code=str(status_code)).time():
            return FastAPIResponse(content=json.dumps({"data": dates}), media_type="application/json", status_code=status_code)
    except Exception as e:
        status_code = 500
        REQUEST_COUNTER.labels(endpoint="/getdatelist", status_code=str(status_code)).inc()
        with REQUEST_LATENCY.labels(endpoint="/getdatelist", status_code=str(status_code)).time():
            return FastAPIResponse(content=f'{{"error": "{str(e)}"}}', media_type="application/json", status_code=status_code)


@app.get("/getclusternames")
async def get_cluster_names(dt_str: str):
    """
    Get cluster names for a given date(yyyy-mm-dd).
    """
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d").date()

        # Now you can use strftime
        formatted_date = dt.strftime("%Y-%m-%d")
    except ValueError:
        status_code = 400
        REQUEST_COUNTER.labels(endpoint="/getclusternames", status_code=str(status_code)).inc()
        with REQUEST_LATENCY.labels(endpoint="/getclusternames", status_code=str(status_code)).time():
            return FastAPIResponse(content='{"error": "Invalid date format"}', media_type="application/json", status_code=status_code)

    news = await get_news_by_date(date=dt)
    if news is None:
        status_code = 404
        REQUEST_COUNTER.labels(endpoint="/getclusternames", status_code=str(status_code)).inc()
        with REQUEST_LATENCY.labels(endpoint="/getclusternames", status_code=str(status_code)).time():
            return FastAPIResponse(content='{"error": "No news found"}', media_type="application/json", status_code=status_code)

    status_code = 200
    REQUEST_COUNTER.labels(endpoint="/getclusternames", status_code=str(status_code)).inc()
    with REQUEST_LATENCY.labels(endpoint="/getclusternames", status_code=str(status_code)).time():
        return FastAPIResponse(content=GetClusterNames(data=news).model_dump_json(), media_type="application/json", status_code=status_code)
    
@app.get("/metrics")
def metrics():
    return FastAPIResponse(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

