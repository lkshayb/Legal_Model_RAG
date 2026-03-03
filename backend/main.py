from fastapi import FastAPI
from pydantic import BaseModel
from query import askQuestion

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(data: Query):
    return askQuestion(data.question)