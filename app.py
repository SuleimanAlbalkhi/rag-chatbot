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
st.caption("Stelle Fragen zu deinen PDF-Dokumenten – lokal & kostenlos.")

try:
    _store_empty = not VECTOR_STORE_DIR.exists() or not any(VECTOR_STORE_DIR.iterdir())
except OSError:
    _store_empty = True

if _store_empty:
    st.error("Kein Vector Store gefunden. Führe zuerst `python ingest.py` aus.")
    st.stop()

if "pipeline" not in st.session_state:
    with st.spinner("Lade Modell..."):
        try:
            st.session_state.pipeline = RAGPipeline()
        except Exception as e:
            st.error(f"Fehler beim Laden der Pipeline: {e}\nIst Ollama gestartet?")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown("**Modell:** Llama 3.2 (lokal)")
    st.markdown("**Embeddings:** nomic-embed-text")
    st.markdown("**Vector Store:** ChromaDB")
    st.divider()
    if st.button("🗑️ Gespräch zurücksetzen"):
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
            with st.expander("📚 Quellen anzeigen"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"**Quelle {i}** – `{src['file']}`, Seite `{src['page']}`")
                    st.caption(src["content"])
                    st.divider()

user_input = st.chat_input("Stelle eine Frage zu deinen Dokumenten...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Denke nach..."):
            try:
                result = st.session_state.pipeline.ask(user_input)
            except Exception as e:
                logging.exception("Unerwarteter Fehler in pipeline.ask()")
                result = {"answer": "Ein unerwarteter Fehler ist aufgetreten. Bitte versuche es erneut.", "sources": []}

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📚 Quellen anzeigen"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**Quelle {i}** – `{src['file']}`, Seite `{src['page']}`")
                    st.caption(src["content"])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
