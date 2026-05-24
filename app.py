import logging
import streamlit as st
from rag_pipeline import RAGPipeline
from config import VECTOR_STORE_DIR

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.caption("Ask questions about your PDF documents – local & free.")

try:
    _store_empty = not VECTOR_STORE_DIR.exists() or not any(VECTOR_STORE_DIR.iterdir())
except OSError:
    _store_empty = True

if _store_empty:
    st.error("No vector store found. Run `python ingest.py` first.")
    st.stop()

if "pipeline" not in st.session_state:
    with st.spinner("Loading model..."):
        try:
            st.session_state.pipeline = RAGPipeline()
        except Exception as e:
            st.error(f"Error loading pipeline: {e}\nIs Ollama running?")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown("**Model:** Llama 3.2 (local)")
    st.markdown("**Embeddings:** nomic-embed-text")
    st.markdown("**Vector Store:** ChromaDB")
    st.divider()
    if st.button("🗑️ Reset conversation"):
        st.session_state.messages = []
        st.session_state.pipeline.chat_history = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.text(message["content"])
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Show sources"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}** – `{src['file']}`, Page `{src['page']}`")
                    st.caption(src["content"])
                    st.divider()

user_input = st.chat_input("Ask a question about your documents...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.pipeline.ask(user_input)
            except Exception as e:
                logging.exception("Unexpected error in pipeline.ask()")
                result = {"answer": "An unexpected error occurred. Please try again.", "sources": []}

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📚 Show sources"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {i}** – `{src['file']}`, Page `{src['page']}`")
                    st.caption(src["content"])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
