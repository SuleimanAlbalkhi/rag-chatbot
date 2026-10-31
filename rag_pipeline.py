import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
import config
from models import get_embeddings, get_llm, get_vector_store

logger = logging.getLogger(__name__)

# The model is told to emit this token instead of a free-text refusal, so refusals
# can be detected reliably regardless of the answer language.
NO_ANSWER_TOKEN = "NO_ANSWER"
REFUSAL_MESSAGE = "I found no information about that in the documents."

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using the context below.

If the context contains an answer, write a clear and concise response based on it.
If the context does not contain relevant information, respond with exactly: NO_ANSWER

Context:
{context}

Chat history:
{chat_history}

Question: {question}
Respond in {language}.
Answer:
"""

_GERMAN_CHARS = set("äöüÄÖÜß")
_GERMAN_WORDS = {
    "was", "ist", "sind", "wie", "warum", "welche", "welcher", "welches",
    "der", "die", "das", "und", "von", "mit", "bei", "erkläre", "nenne",
    "beschreibe", "zeige", "gib", "mir", "kannst", "du", "bitte", "welchen",
}

History = Sequence[tuple[str, str]]


class EmptyVectorStoreError(RuntimeError):
    """The vector store is missing or contains no chunks."""


@dataclass(frozen=True)
class RetrievedChunk:
    id: str | None
    file: str
    page: int | None  # 1-based physical page number
    page_label: str | None  # label printed in the PDF, may differ from `page`
    text: str
    score: float


@dataclass(frozen=True)
class Result:
    answer: str
    refused: bool
    sources: list[RetrievedChunk] = field(default_factory=list)  # chunks cited for the answer; empty on refusal
    chunks: list[RetrievedChunk] = field(default_factory=list)  # everything retrieved, with scores
    error: str | None = None


class RAGPipeline:
    """Stateless RAG core: every call to answer() is independent; callers own the history."""

    def __init__(self, vector_store: VectorStore | None = None, llm: BaseChatModel | None = None):
        if vector_store is None:
            # Check before Chroma touches the path: it would silently create an empty store.
            if not config.VECTOR_STORE_DIR.exists():
                raise EmptyVectorStoreError(
                    f"No vector store at '{config.VECTOR_STORE_DIR}'. Run `python ingest.py` first."
                )
            vector_store = get_vector_store(get_embeddings())
        if not vector_store.get(limit=1)["ids"]:
            raise EmptyVectorStoreError(
                f"Vector store collection '{config.COLLECTION_NAME}' at '{config.VECTOR_STORE_DIR}' "
                "is empty. Run `python ingest.py --reset`."
            )
        self.vector_store = vector_store
        self.llm = llm or get_llm()
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm

    def _detect_language(self, text: str) -> str:
        if any(c in _GERMAN_CHARS for c in text):
            return "German"
        words = set(text.lower().split())
        if words & _GERMAN_WORDS:
            return "German"
        return "English"

    @staticmethod
    def _format_history(history: History) -> str:
        if not history:
            return "No prior conversation."
        lines = []
        for human, ai in history:
            lines.append(f"User: {human}")
            lines.append(f"Assistant: {ai}")
        return "\n".join(lines)

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Top-k chunks whose relevance score reaches the configured threshold."""
        scored = self.vector_store.similarity_search_with_relevance_scores(question, k=config.TOP_K)
        return [
            RetrievedChunk(
                id=doc.id,
                file=Path(doc.metadata.get("source", "unknown")).name,
                page=doc.metadata.get("page"),
                page_label=doc.metadata.get("page_label"),
                text=doc.page_content,
                score=score,
            )
            for doc, score in scored
            if score >= config.SCORE_THRESHOLD
        ]

    def answer(self, question: str, history: History | None = None) -> Result:
        if not question.strip():
            return Result(answer="Please ask a question.", refused=False, error="empty_question")

        try:
            chunks = self.retrieve(question)
        except Exception as e:
            logger.exception("Retrieval failed")
            return Result(answer="", refused=False, error=f"retrieval_failed: {e}")

        if not chunks:
            return Result(answer=REFUSAL_MESSAGE, refused=True)

        recent = list(history or [])[-config.MAX_HISTORY_TURNS:]
        try:
            response = self.chain.invoke({
                "context": "\n\n".join(c.text for c in chunks),
                "chat_history": self._format_history(recent),
                "question": question,
                "language": self._detect_language(question),
            })
        except Exception as e:
            logger.exception("Generation failed")
            return Result(answer="", refused=False, chunks=chunks, error=f"generation_failed: {e}")

        text = str(response.content).strip()
        if NO_ANSWER_TOKEN in text:
            return Result(answer=REFUSAL_MESSAGE, refused=True, chunks=chunks)

        return Result(answer=text, refused=False, sources=chunks, chunks=chunks)
