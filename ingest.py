import argparse
import shutil
import sys
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import config
from models import ModelUnavailableError, get_embeddings


def load_pdf(pdf) -> list[Document]:
    reader = PdfReader(str(pdf))
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": str(pdf), "page": i})
        for i, page in enumerate(reader.pages)
    ]


def load_pdfs():
    docs = []
    pdf_files = list(config.DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{config.DOCUMENTS_DIR}'!")
    for pdf in pdf_files:
        print(f"Loading: {pdf.name}")
        try:
            docs.extend(load_pdf(pdf))
        except Exception as e:
            print(f"Warning: {pdf.name} could not be loaded: {e}")
    print(f"{len(docs)} pages loaded.")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} chunks created.")
    return chunks


def build_vector_store(chunks):
    print("Creating embeddings and storing in ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTOR_STORE_DIR)
    )
    print("Vector store saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB vector store.")
    parser.add_argument("--reset", action="store_true", help="Delete existing vector store before ingesting.")
    args = parser.parse_args()

    # Fail before touching the existing store if Ollama or the embedding model is unavailable.
    try:
        get_embeddings()
    except ModelUnavailableError as e:
        sys.exit(f"Error: {e}")

    if config.VECTOR_STORE_DIR.exists() and any(config.VECTOR_STORE_DIR.iterdir()):
        if args.reset:
            print("Deleting existing vector store...")
            shutil.rmtree(config.VECTOR_STORE_DIR)
        else:
            print(
                f"Vector store already exists at '{config.VECTOR_STORE_DIR}'.\n"
                "Use --reset to overwrite it: python ingest.py --reset"
            )
            sys.exit(0)

    docs = load_pdfs()
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
