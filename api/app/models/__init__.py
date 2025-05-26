from pydantic import BaseModel
from datetime import datetime

class News(BaseModel):
    title: str
    content: str
    date: datetime
    cluster: int