"""
ingest.py
Reads all .txt files from docs/, splits them into chunks, embeds them
using a local Ollama embedding model, and stores them in a persistent
Chroma vector database (./chroma_db).

Run this once whenever you add/change files in docs/:
    python ingest.py
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DOCS_DIR = "docs"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"  # run: ollama pull nomic-embed-text


def load_documents():
    """Load every .txt file in docs/ as a LangChain Document, keeping
    the filename as metadata so retrieved answers can cite their source."""
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents


def main():
    print("Loading documents from docs/ ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Embedding chunks using '{EMBEDDING_MODEL}' (this calls your local Ollama instance) ...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(f"Building/persisting Chroma vector store at ./{PERSIST_DIR} ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print("Done. Vector store is ready.")
    print(f"Total vectors stored: {vectorstore._collection.count()}")


if __name__ == "__main__":
    main()