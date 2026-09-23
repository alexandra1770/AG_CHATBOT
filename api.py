from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from chatbot_core import answer_question


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

    answer, sources = answer_question(
        data.question,
        data.history
    )

    source_list = []

    for doc in sources:
        source_list.append({
            "file": doc.metadata.get("source_file"),
            "page": doc.metadata.get("page")
        })

    return {
        "question": data.question,
        "answer": answer,
        "sources": source_list
    }
