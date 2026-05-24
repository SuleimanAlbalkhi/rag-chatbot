from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import VECTOR_STORE_DIR

MAX_HISTORY = 10
SIMILARITY_THRESHOLD = 0.50

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using the context below.

If the context contains an answer, write a clear and concise response based on it.
If the context does not contain relevant information, respond: "I found no information about that in the documents."

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


class RAGPipeline:
    def __init__(self):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": SIMILARITY_THRESHOLD, "k": 10},
        )
        self.llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm
        self.chat_history = []

    def _detect_language(self, text: str) -> str:
        if any(c in _GERMAN_CHARS for c in text):
            return "German"
        words = set(text.lower().split())
        if words & _GERMAN_WORDS:
            return "German"
        return "English"

    def _format_history(self) -> str:
        if not self.chat_history:
            return "No prior conversation."
        lines = []
        for human, ai in self.chat_history:
            lines.append(f"User: {human}")
            lines.append(f"Assistant: {ai}")
        return "\n".join(lines)

    def ask(self, question: str) -> dict:
        if not question.strip():
            return {"answer": "Please ask a question.", "sources": []}

        try:
            docs = self.retriever.invoke(question)
        except Exception as e:
            return {"answer": f"Error retrieving documents: {e}", "sources": []}

        if not docs:
            return {"answer": "I found no information about that in the documents.", "sources": []}

        context = "\n\n".join(doc.page_content for doc in docs)

        try:
            response = self.chain.invoke({
                "context": context,
                "chat_history": self._format_history(),
                "question": question,
                "language": self._detect_language(question),
            })
        except Exception as e:
            return {"answer": f"Error generating response: {e}", "sources": []}

        answer = str(response.content) if hasattr(response, "content") else str(response)
        self.chat_history.append((question, answer))
        self.chat_history = self.chat_history[-MAX_HISTORY:]

        seen = set()
        sources = []
        for doc in docs:
            file = Path(doc.metadata.get("source", "unknown")).name
            page = doc.metadata.get("page", "?")
            key = (file, page)
            if key not in seen:
                seen.add(key)
                sources.append({"file": file, "page": page, "content": doc.page_content[:300]})

        return {"answer": answer, "sources": sources}
