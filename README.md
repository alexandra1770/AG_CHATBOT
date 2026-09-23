\# Genetic Algorithms RAG Chatbot



This project is an AI chatbot that I developed for answering questions about Genetic Algorithms.



The chatbot uses my Genetic Algorithms course and laboratory materials as its knowledge base. I started the project as a RAG chatbot and then extended it with an AI Agent, tools and MCP (Model Context Protocol).



The main idea was to build a chatbot that does not only generate an answer, but can also decide when it needs to search the course documents or use another tool.



\## How it works



The application has the following flow:



```text

User

&#x20; ↓

Gradio

&#x20; ↓

FastAPI

&#x20; ↓

AI Agent

&#x20; ↓

MCP Client

&#x20; ↓

MCP Server

&#x20; ↓

Tools

&#x20; ├── search\_documents

&#x20; └── calculator

```



For questions about Genetic Algorithms, the Agent can use `search\_documents` to retrieve information from the course materials.



For mathematical calculations, it can use the `calculator` tool.



After receiving the result from the tool, the Agent generates the final answer.



\## Technologies used



\- Python

\- LangChain

\- Groq

\- ChromaDB

\- Hugging Face Sentence Transformers

\- MCP

\- FastAPI

\- Gradio

\- pytest



The embedding model used for the documents is:



```text

sentence-transformers/paraphrase-multilingual-mpnet-base-v2

```



The LLM currently used by the Agent is:



```text

openai/gpt-oss-20b

```



\## RAG



The RAG part of the project retrieves information from Genetic Algorithms course and laboratory documents.



The basic flow is:



```text

Question

&#x20;  ↓

Document retrieval

&#x20;  ↓

Reranking

&#x20;  ↓

Relevant documents

&#x20;  ↓

Agent

&#x20;  ↓

Answer

```



The retrieved documents also contain metadata such as the source file and page.



\## Agent and Tools



I added an Agent so the chatbot can decide which tool it needs.



At the moment there are two tools:



\### search\_documents



Searches the Genetic Algorithms documents and returns the most relevant information.



\### calculator



Used when an exact mathematical calculation is needed.



For example:



```text

Question: Calculate 25 \* 17



Agent

&#x20; ↓

calculator

&#x20; ↓

425

&#x20; ↓

Final answer

```



The Agent also has a maximum number of steps and protection against calling the document search repeatedly for the same question.



\## MCP



I used Model Context Protocol to separate the Agent from the tools.



```text

Agent

&#x20; ↓

MCP Client

&#x20; ↓

MCP Server

&#x20; ↓

Tool

```



The MCP server currently exposes:



```text

search\_documents

calculator

```



This was also useful for learning how an Agent can communicate with tools through MCP instead of calling every function directly.



\## Conversation history



The chatbot keeps the previous messages and sends them to the Agent.



This allows follow-up questions such as:



```text

What is mutation?



Explain it more simply.



Give me an example.

```



\## Evaluation



I also added several ways to test the system.



\### RAG evaluation



I created a small manually checked evaluation set for the document retrieval system.



The metrics used are:



\- Recall@5

\- Top-1 Accuracy

\- MRR



At the moment this evaluation contains 5 manually validated questions, so the results are only for this small test set.



\### Answer evaluation



There is also an evaluation script for checking generated answers, including expected concepts and source references.



API errors such as rate limits are treated separately from incorrect answers.



\### Groundedness evaluation



I implemented a basic groundedness evaluator to check whether an answer is supported by the retrieved context.



It currently uses:



\- lexical similarity

\- semantic similarity

\- best matching context sentence

\- negation checking



This is a basic evaluation method and not a complete hallucination detection system.



\## Tests



I used `pytest` for automated testing.



The tests cover:



\- Agent tool usage

\- MCP

\- calculator

\- document search

\- groundedness



Current result:



```text

13 passed

```



The tests are located in:



```text

tests/

├── test\_agent.py

├── test\_groundedness.py

├── test\_mcp.py

└── test\_tools.py

```



\## Logging



I added logging to see what the Agent is doing while processing a question.



The application can log:



\- the question

\- selected tool

\- Agent step

\- MCP success/error

\- repeated searches

\- execution time



The logs are saved in:



```text

logs/agent.log

```



\## API and interface



The backend is implemented with FastAPI.



The main endpoint is:



```text

POST /ask

```



The user interface is implemented with Gradio.



It supports Romanian and English questions, conversation history, Markdown and mathematical formulas.



\## Project structure



```text

AG\_Chatbot/

│

├── api.py

├── chatbot\_core.py

├── gradio\_app.py

├── logger\_config.py

├── mcp\_server.py

├── mcp\_client.py

│

├── evaluation/

│   ├── rag\_evaluation.py

│   ├── answer\_evaluation.py

│   └── groundedness\_evaluation.py

│

├── tests/

│   ├── test\_agent.py

│   ├── test\_groundedness.py

│   ├── test\_mcp.py

│   └── test\_tools.py

│

└── README.md

```



\## What I learned from this project



Through this project I worked with:



\- RAG systems

\- embeddings and vector databases

\- LLM tool calling

\- AI Agents

\- MCP servers and clients

\- FastAPI

\- Gradio

\- automated testing

\- RAG and answer evaluation

\- logging and error handling



\## Next steps



Some things I would like to add or improve:



\- more evaluation questions

\- better groundedness evaluation

\- more MCP tools

\- Docker

\- deployment

\- improved monitoring

