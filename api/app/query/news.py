from prisma import Prisma
from prisma.models import News
from prisma.errors import PrismaError
from typing import List, Optional

from datetime import datetime, timedelta
from redis import Redis

redis = Redis(host='redis', port=6379, db=0,decode_responses=True)

__all__ = ["query_news", "store_news"]

def get_start_of_week(date: datetime) -> datetime:
    start_of_week = date - timedelta(days=date.weekday())
    
    return start_of_week

def date_parser(dt: datetime) -> datetime:
    """
    Parse a datetime object to a date object.
    """
    
    return datetime.combine(dt.date(), datetime.min.time())

async def query_news(date: Optional[datetime] = None, test: Optional[bool]= False) -> Optional[News]:
    """
    Fetch news articles from the database.
    If a date is provided, it will return news articles for that date.
    Otherwise, it will return the latest news articles.
    """
    
    if date is None:
        date = datetime.now()
    
    news_article = redis.get(date.isoformat())
    if news_article:
        print(f"Found in Redis: {news_article}")
        return News.model_validate_json(news_article)

    db = Prisma()
    await db.connect()
    
    if test:
        news_article = await db.newstest.find_first(order={"id": "desc"})
        
        await db.disconnect()
        return news_article
    print(f"Date: {date}")
    if date is None:
        news_article = await db.news.find_first(order={"id": "desc"})
    else:
        start_of_week = get_start_of_week(date)
        print(f"Start of week: {start_of_week} of {date}")
        
        news_article = await db.news.find_first(
            where={
                "startDate": date_parser(start_of_week),
            },
        )
    
    await db.disconnect()
    return news_article

async def store_news(
    title: str,
    content: str,
    date: datetime,
) -> bool:
    """
    Store a news article in the database.
    """
    db = Prisma()
    
    await db.connect()
    
    try:
        news_article = await db.news.create(
            data={
                "title": title,
                "content": content,
                "startDate": date_parser(get_start_of_week(date)),
            },
        )
        
        await db.disconnect()
        
        redis.set(news_article.startDate.isoformat(), news_article.model_dump_json(), ex=60*60*24*14)
        
        return True if news_article else False
    except PrismaError as e:
        print(f"Error storing news article: {e}")
        await db.disconnect()
        return False