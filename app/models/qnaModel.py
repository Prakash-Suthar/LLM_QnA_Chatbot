from pydantic import BaseModel

class QnAModel(BaseModel):
    query: str
