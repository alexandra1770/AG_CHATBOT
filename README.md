# AG_CHATBOT
Answers questions with answers from university courses about Genetic Algorithms

# Genetic Algorithms Assistant

A bilingual **RAG-based AI chatbot** designed to answer questions about Genetic Algorithms using university course and laboratory materials.

The assistant supports both **Romanian and English**, maintains conversation context, retrieves relevant information from the course materials and can generate Python code for Genetic Algorithms problems.

## Features

- Romanian and English support
- Answers based on Genetic Algorithms course materials
- Hybrid semantic and lexical retrieval
- LLM-based document reranking
- Conversational follow-up questions
- Python code generation
- LaTeX mathematical formulas
- Source file and page references

## Technologies

- Python
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Groq LLM
- FastAPI
- Gradio

## Project Structure

```text
AG_Chatbot/
│
├── assets/
│   ├── main_interface.png
│   ├── romanian_answer.png
│   ├── english_answer.png
│   └── code_generation.png
│
├── api.py
├── chatbot_core.py
├── gradio_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Next Step

The next planned improvement is the integration of **Tavily Web Search**.

This will allow the assistant to combine the existing local RAG system with information retrieved from the web, while keeping the university course materials as the primary knowledge source.

The planned workflow is:

```text
User Question
      |
      v
Local RAG
      |
      v
Course Materials
      +
Tavily Web Search
      |
      v
Groq LLM
      |
      v
Final Answer
```

This extension will allow the chatbot to provide additional information and external sources when relevant.
