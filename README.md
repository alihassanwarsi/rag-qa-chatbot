# RAG QA Chatbot

A simple RAG-based chatbot that lets you ask questions about a PDF document using LangChain, ChromaDB, Hugging Face embeddings, Google Gemini, and Streamlit.

## Tech Stack

* Python
* LangChain
* Google Gemini
* Hugging Face
* ChromaDB
* Streamlit
* PyPDF

## Features

* PDF question answering
* Retrieval-Augmented Generation (RAG)
* Semantic search with `all-MiniLM-L6-v2`
* ChromaDB vector storage
* Google Gemini for answer generation
* Streamlit chat interface
* Persistent vector database

## Project Structure

```text
rag-qa-chatbot/
├── AI.pdf
├── app.py
├── bot.py
├── requirements.txt
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
└── README.md
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
```

## Run

```bash
streamlit run app.py
```

## Example Questions

```text
What is retrieval?
What is the main topic of the document?
Explain the concept discussed in the document.
```

## How It Works

```text
AI.pdf
   ↓
PDF Loader
   ↓
Text Splitting
   ↓
Hugging Face Embeddings
   ↓
ChromaDB
   ↓
Retriever
   ↓
Google Gemini
   ↓
Response
   ↓
Streamlit
```

1. Load the PDF using `PyPDFLoader`
2. Split the document into chunks
3. Generate embeddings using `all-MiniLM-L6-v2`
4. Store the embeddings in ChromaDB
5. Retrieve relevant chunks for a question
6. Send the retrieved context to Gemini
7. Display the generated answer in Streamlit

## Models

**LLM:** `gemini-3.5-flash-lite`

**Embedding Model:** `all-MiniLM-L6-v2`

## About

A beginner RAG project built to understand document loading, chunking, embeddings, vector databases, retrieval, and LLM-based question answering.
