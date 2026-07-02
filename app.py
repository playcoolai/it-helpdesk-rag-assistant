"""
app.py
Streamlit chat UI for the IT Helpdesk Knowledge Assistant.

Run with:
    streamlit run app.py

Prerequisite: run `python ingest.py` at least once first to build the
vector store, and make sure Ollama is running with the required models
pulled (llama3, nomic-embed-text).
"""

import streamlit as st
from rag_engine import build_rag_chain

st.set_page_config(page_title="IT Helpdesk Knowledge Assistant", page_icon="🛠️", layout="centered")

st.title("🛠️ IT Helpdesk Knowledge Assistant")
st.caption(
    "Ask a question about AD, DNS/DHCP, RDS, backups, NTFS permissions, "
    "network issues, printers, or vendor/AMC procedures. "
    "Answers are grounded in the internal knowledge base — powered entirely "
    "by a local, open-source LLM (no cloud API, no per-query cost)."
)


@st.cache_resource
def get_chain_and_retriever():
    return build_rag_chain()


try:
    chain, retriever = get_chain_and_retriever()
except Exception as e:
    st.error(
        "Could not load the knowledge base. Have you run `python ingest.py` yet, "
        "and is Ollama running with the required models pulled?\n\n"
        f"Details: {e}"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask an IT support question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the knowledge base..."):
            answer = chain.invoke(user_question)
            retrieved_docs = retriever.invoke(user_question)
            sources = sorted(set(doc.metadata.get("source", "unknown") for doc in retrieved_docs))

        st.markdown(answer)
        if sources:
            st.caption("📄 Sources referenced: " + ", ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **IT Helpdesk Knowledge Assistant**

        A Retrieval-Augmented Generation (RAG) tool that answers IT
        support questions strictly from internal documentation, running
        entirely on local, open-source infrastructure:

        - **LLM**: Llama 3 (via Ollama)
        - **Embeddings**: nomic-embed-text (via Ollama)
        - **Vector store**: ChromaDB
        - **Orchestration**: LangChain
        - **UI**: Streamlit

        No cloud API calls, no per-query cost, and all company data stays
        on-premise — built for small/mid-size businesses that can't afford
        (or don't want) to send internal documentation to a third-party API.
        """
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()