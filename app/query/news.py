from prisma import Prisma
from prisma.models import News
from prisma.errors import PrismaError
from typing import List, Optional

from datetime import datetime, timedelta

__all__ = ["query_news"]

def get_start_of_week(date: datetime) -> datetime:
    # Get the start of the week (Monday)
    start_of_week = date - timedelta(days=date.weekday())
    
    # return only the date part
    return start_of_week

def date_parser(datetime: datetime) -> datetime:
    """

    """
    return datetime.combine(datetime.date(), datetime.min.time())

async def query_news(date: Optional[datetime] = None, test: Optional[bool]= False) -> Optional[News]:
    """
    Fetch news articles from the database.
    If a date is provided, it will return news articles for that date.
    Otherwise, it will return the latest news articles.
    """
    db = Prisma()
        
    await db.connect()
    
    if test:
        news_article = await db.newstest.find_first(order={"id": "desc"})
        
        await db.disconnect()
        return news_article
    print(f"Date: {date}")
    if date is None:
        # Get the latest news articles
        news_article = await db.news.find_first(order={"id": "desc"})
    else:
        # Get the start of the week (Monday)
        start_of_week = get_start_of_week(date)
        print(f"Start of week: {start_of_week} of {date}")
        
        # Get news articles for the specified date
        news_article = await db.news.find_first(
            where={
                "startDate": date_parser(start_of_week),
            },
        )
    
    
    await db.disconnect()
    return news_article
