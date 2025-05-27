from pydantic import BaseModel
from datetime import datetime
from typing import List 
from prisma.models import News as PrismaNews

class News(BaseModel):
    title: str
    content: str
    date: datetime
    cluster: int
    
class GetClusterNames(BaseModel):
    data: List[PrismaNews]