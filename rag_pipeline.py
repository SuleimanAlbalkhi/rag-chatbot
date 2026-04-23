from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

VECTOR_STORE_DIR = Path("vector_store")

PROMPT_TEMPLATE = """
You are a helpful assistant answering questions about uploaded documents.
Use the provided context to answer. If the context covers the topic partially, 
summarize what you find. Only say you don't know if the context is completely unrelated.

Context:
{context}

Chat history:
{chat_history}

Question: {question}
Answer:
"""

class RAGPipeline:
    def __init__(self):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        self.llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chat_history = []  # Einfache Liste statt Memory-Klasse

    def _format_history(self) -> str:
        if not self.chat_history:
            return "Kein bisheriges Gespräch."
        lines = []
        for human, ai in self.chat_history:
            lines.append(f"Nutzer: {human}")
            lines.append(f"Assistent: {ai}")
        return "\n".join(lines)

    def ask(self, question: str) -> dict:
        if not question.strip():
            return {"answer": "Bitte stelle eine Frage.", "sources": []}

        docs = self.retriever.invoke(question)
        if not docs:
            return {"answer": "Keine relevanten Informationen gefunden.", "sources": []}

        context = "\n\n".join(doc.page_content for doc in docs)
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "chat_history": self._format_history(),
            "question": question
        })

        self.chat_history.append((question, response.content))

        sources = [
            {
                "file": Path(doc.metadata.get("source", "unbekannt")).name,
                "page": doc.metadata.get("page", "?"),
                "content": doc.page_content[:300]
            }
            for doc in docs
        ]
        return {"answer": response.content, "sources": sources}