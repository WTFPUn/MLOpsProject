from typing import Optional
from datetime import date
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response as FastAPIResponse
from prometheus_fastapi_instrumentator import Instrumentator

from query import query_news

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
        return FastAPIResponse(content=news.model_dump_json(), media_type="application/json", status_code=status_code)

@app.get("/metrics")
def metrics():
    return FastAPIResponse(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
