import streamlit as st
from rag_pipeline import RAGPipeline, VectorStoreError
from models import ModelUnavailableError
from safe_markdown import neutralize_links

ERROR_MESSAGE = "Sorry, something went wrong while answering. Please try again."

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.caption("Ask questions about your PDF documents – local & free.")


@st.cache_resource(show_spinner="Loading model...")
def load_pipeline() -> RAGPipeline:
    # The pipeline is stateless, so one instance is shared by all sessions.
    return RAGPipeline()


try:
    pipeline = load_pipeline()
except (VectorStoreError, ModelUnavailableError) as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error loading pipeline: {e}\nIs Ollama running?")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def chat_history() -> list[tuple[str, str]]:
    """(question, answer) pairs from this session, skipping turns that failed."""
    msgs = st.session_state.messages
    return [
        (q["content"], a["content"])
        for q, a in zip(msgs[::2], msgs[1::2])
        if not a.get("error")
    ]


def page_ref(src) -> str:
    ref = f"Page {src.page}"
    if src.page_label and src.page_label != str(src.page):
        ref += f" (printed: {src.page_label})"
    return ref


def render_sources(sources):
    if not sources:
        return
    with st.expander("📚 Show sources"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**Source {i}** – `{src.file}`, {page_ref(src)} · relevance {src.score:.2f}")
            st.caption(neutralize_links(src.text))
            st.divider()


with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown("**Model:** Llama 3.2 (local)")
    st.markdown("**Embeddings:** nomic-embed-text")
    st.markdown("**Vector Store:** ChromaDB")
    st.divider()
    if st.button("🗑️ Reset conversation"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.text(message["content"])
        else:
            st.markdown(ERROR_MESSAGE if message.get("error") else neutralize_links(message["content"]))
            render_sources(message.get("sources"))

user_input = st.chat_input("Ask a question about your documents...")
if user_input:
    history = chat_history()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = pipeline.answer(user_input, history)
        st.markdown(ERROR_MESSAGE if result.error else neutralize_links(result.answer))
        render_sources(result.sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "sources": result.sources,
        "error": result.error,
    })
