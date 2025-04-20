from typing import Union, Optional
from datetime import datetime, timedelta, date
from fastapi import FastAPI, Request, Response

from query import query_news

app = FastAPI()


@app.get("/health")
async def health_check() -> Response:
    """
    Health check endpoint to verify if the service is running.
    Returns a JSON response with the status and HTTP status code.
    """
    return Response(content='{"status": "ok"}', media_type="application/json", status_code=200)


@app.get("/news")
async def get_news(date: Optional[date] = None, test: Optional[bool]=False) :
    """
    Endpoint to get news articles.
    If a date is provided, it will return news articles for that date.
    Otherwise, it will return the latest news articles.
    """
    
    news = await query_news(date, test)
    
    if news is None:
        return Response(content='{"error": "No news found"}', media_type="application/json", status_code=404)
    
    return Response(content=news.model_dump_json(), media_type="application/json", status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
