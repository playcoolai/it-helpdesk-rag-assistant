"""
rag_engine.py
Core retrieval-augmented generation logic. Loads the persisted Chroma
vector store built by ingest.py, retrieves relevant chunks for a user
question, and asks a local Ollama model to answer strictly using that
retrieved context - citing which source document it came from.
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"  # swap to "qwen2.5:7b" if you pull it later

SYSTEM_PROMPT = """You are an internal IT Helpdesk Assistant for a small/mid-size business.
Answer the user's question using ONLY the context provided below, which comes from the
company's internal IT documentation (SOPs, troubleshooting guides, procedures).

Rules:
- If the answer is not contained in the context, say clearly: "I don't have that information
  in the current knowledge base." Do not guess or make up an answer.
- Keep answers practical and step-by-step where the source material is step-by-step.
- At the end of your answer, list which source document(s) you used, based on the
  "Source:" labels in the context.

Context:
{context}
"""


def format_docs(docs):
    """Format retrieved chunks with their source filename visible to the model."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def get_answer_with_sources(question: str):
    """Convenience function: returns (answer_text, list_of_source_filenames)."""
    chain, retriever = build_rag_chain()
    answer = chain.invoke(question)
    retrieved_docs = retriever.invoke(question)
    sources = sorted(set(doc.metadata.get("source", "unknown") for doc in retrieved_docs))
    return answer, sources


if __name__ == "__main__":
    # Quick command-line test without Streamlit
    test_question = "A user's account keeps getting locked out, what should I check?"
    answer, sources = get_answer_with_sources(test_question)
    print("Q:", test_question)
    print("\nA:", answer)
    print("\nSources used:", sources)