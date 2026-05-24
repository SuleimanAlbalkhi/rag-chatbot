import argparse
import shutil
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCUMENTS_DIR, VECTOR_STORE_DIR


def load_pdfs():
    docs = []
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDFs found in the 'documents/' folder!")
    for pdf in pdf_files:
        print(f"Loading: {pdf.name}")
        try:
            loader = PyPDFLoader(str(pdf))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: {pdf.name} could not be loaded: {e}")
    print(f"{len(docs)} pages loaded.")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} chunks created.")
    return chunks


def build_vector_store(chunks):
    print("Creating embeddings and storing in ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )
    print("Vector store saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB vector store.")
    parser.add_argument("--reset", action="store_true", help="Delete existing vector store before ingesting.")
    args = parser.parse_args()

    if VECTOR_STORE_DIR.exists() and any(VECTOR_STORE_DIR.iterdir()):
        if args.reset:
            print("Deleting existing vector store...")
            shutil.rmtree(VECTOR_STORE_DIR)
        else:
            print(
                f"Vector store already exists at '{VECTOR_STORE_DIR}'.\n"
                "Use --reset to overwrite it: python ingest.py --reset"
            )
            sys.exit(0)

    docs = load_pdfs()
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
