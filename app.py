import streamlit as st
from rag_pipeline import RAGPipeline
from pathlib import Path

VECTOR_STORE_DIR = Path("vector_store")

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="",
    layout="wide"
)

st.title(" RAG Chatbot")
st.caption("Stelle Fragen zu deinen PDF-Dokumenten – lokal & kostenlos.")

# Vector Store prüfen
if not VECTOR_STORE_DIR.exists() or not any(VECTOR_STORE_DIR.iterdir()):
    st.error("Kein Vector Store gefunden. Führe zuerst `python ingest.py` aus.")
    st.stop()

# Pipeline einmalig laden (nicht bei jeder Nutzereingabe neu starten)
if "pipeline" not in st.session_state:
    with st.spinner("Lade Modell..."):
        st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar mit Infos
with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown("**Modell:** Llama 3.2 (lokal)")
    st.markdown("**Embeddings:** nomic-embed-text")
    st.markdown("**Vector Store:** ChromaDB")
    st.divider()
    if st.button("🗑️ Gespräch zurücksetzen"):
        st.session_state.messages = []
        del st.session_state.pipeline
        st.rerun()

# Chatverlauf anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Quellen anzeigen"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"**Quelle {i}** – `{src['file']}`, Seite `{src['page']}`")
                    st.caption(src["content"])
                    st.divider()

# Eingabe
user_input = st.chat_input("Stelle eine Frage zu deinen Dokumenten...")
if user_input:
    # Nutzernachricht anzeigen
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Antwort generieren
    with st.chat_message("assistant"):
        with st.spinner("Denke nach..."):
            result = st.session_state.pipeline.ask(user_input)

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
    
