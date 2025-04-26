from prisma import Prisma
from prisma.models import NewsTest

from datetime import datetime, date
from datetime import timedelta

def get_start_of_week(date: datetime) -> tuple:
    # Get the start of the week (Monday)
    start_of_week = date - timedelta(days=date.weekday())
    
    # return only the date part
    return start_of_week.date()

async def seed():
    db = Prisma(auto_register=True)
    await db.connect()
    
    # Delete all existing news articles
    await db.newstest.delete_many()
     
    start_date_last = datetime.combine(get_start_of_week(datetime(2024,1,1)), datetime.min.time())
    # first is before last 2 months
    start_date_first = datetime.combine(get_start_of_week(datetime(2023,1,1)), datetime.min.time())

    # Create a test news article
    test_news = await db.newstest.create_many(
        data=[
             {
            "title": "First News",
            "content": "This is a test news article.",
            "startDate": start_date_first,
        },
            {
            "title": "Lastest News",
            "content": "This is a test news article.",
            "startDate": start_date_last,
        },
        ],
        

              
    )

    print(f"Created test news article: {test_news}")

    await db.disconnect()
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(seed())