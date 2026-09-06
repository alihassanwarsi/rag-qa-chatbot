import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

prompt_template = """You are a helpful document question-answering assistant.

For casual conversation such as greetings, thanks, or simple small talk, respond naturally.

For questions about the document, use the provided context to answer the question.
If the answer to a document-related question is not found in the context, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

CHROMA_DIR = "./chroma_db"

def load_documents():
    pdf_path = "AI.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def build_vector_store():
    """Load existing Chroma DB if present, otherwise build it from the PDF.
    Skips PDF loading/splitting entirely when the DB already exists."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(f"{CHROMA_DIR}/chroma.sqlite3"):
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        print("Loaded existing Chroma DB.")
    else:
        docs = load_documents()
        chunks = split_text(docs)
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        print("Created and persisted Chroma DB.")

    return db

def create_qa_chain(db):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GEMINI_API_KEY
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def initialize_qa_chain():
    db = build_vector_store()
    return create_qa_chain(db)

def ask_question(qa_chain, query):
    result = qa_chain.invoke({"query": query})

    print("\nAnswer:")
    print(result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"- Page {doc.metadata.get('page', 'N/A')}")

if __name__ == "__main__":
    qa_chain = initialize_qa_chain()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        ask_question(qa_chain, query)