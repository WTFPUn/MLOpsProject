from datetime import datetime, timedelta
from typing import Optional, List
from prisma import Prisma
import asyncio

def get_start_of_week(date: datetime) -> datetime:
    start_of_week = date - timedelta(days=date.weekday())
    
    return start_of_week

def date_parser(dt: datetime) -> datetime:
    """
    Parse a datetime object to a date object.
    """
    
    return datetime.combine(dt, datetime.min.time())

async def main():
    """
    Main function to test the database connection and query.
    """
    db = Prisma()

    date = datetime.now()  # Replace with the desired date or pass it as an argument
        
    await db.connect()

    start_of_week = get_start_of_week(date)

    news_article = await db.news.find_many(
        where={
            "startDate": date_parser(start_of_week),
        },
        select={
            "id": True,
            "title": True,
        },
    )
    
    
    await db.disconnect()
    
    print(f"News articles for the week starting {start_of_week}:")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())