from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from chatbot_core import answer_with_agent


app = FastAPI()


class QuestionRequest(BaseModel):
    question: str
    history: List[Dict[str, Any]] = Field(default_factory=list)


@app.get("/")
def home():
    return {
        "message": "Genetic Algorithms Chatbot API is running"
    }


@app.post("/ask")
def ask_question(data: QuestionRequest):

    answer = answer_with_agent(
        data.question,
        data.history
    )

    return {
        "question": data.question,
        "answer": answer
    }