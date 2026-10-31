import argparse
import shutil
import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import config
from models import ModelUnavailableError, get_embeddings


class IngestError(RuntimeError):
    """Ingestion cannot produce a complete, usable vector store."""


def load_pdf(pdf: Path) -> list[Document]:
    """One Document per page. `page` is the 1-based physical page number (what a PDF viewer
    navigates to); `page_label` is the label printed in the PDF's page-label table, if any."""
    reader = PdfReader(str(pdf))
    labels = reader.page_labels
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": pdf.name, "page": i + 1, "page_label": labels[i]},
        )
        for i, page in enumerate(reader.pages)
    ]


def load_pdfs(skip_failed: bool = False) -> list[Document]:
    pdf_files = sorted(config.DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise IngestError(f"No PDFs found in '{config.DOCUMENTS_DIR}'.")

    docs, failed = [], []
    for pdf in pdf_files:
        try:
            pages = load_pdf(pdf)
        except Exception as e:
            failed.append(f"{pdf.name}: could not be read ({e})")
            continue
        empty = [d.metadata["page"] for d in pages if not d.page_content.strip()]
        if len(empty) == len(pages):
            failed.append(f"{pdf.name}: no extractable text (scanned PDF without OCR?)")
            continue
        print(f"Loaded: {pdf.name} ({len(pages)} pages)")
        if empty:
            print(f"  Warning: {len(empty)} page(s) without text: {empty[:20]}{' ...' if len(empty) > 20 else ''}")
        docs.extend(pages)

    if failed:
        report = "\n".join(f"  - {f}" for f in failed)
        if not skip_failed or not docs:
            raise IngestError(f"{len(failed)} PDF(s) failed:\n{report}\nFix them or rerun with --skip-failed.")
        print(f"Skipping {len(failed)} PDF(s):\n{report}")
    print(f"{len(docs)} pages loaded.")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split pages into chunks and give each a deterministic id: '<file>:p<page>:c<index on page>'."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for page in docs:
        for i, chunk in enumerate(splitter.split_documents([page])):
            chunk.metadata["chunk_index"] = i
            chunk.id = f"{chunk.metadata['source']}:p{chunk.metadata['page']}:c{i}"
            chunks.append(chunk)
    if not chunks:
        raise IngestError("No chunks were created; the PDFs contain no usable text.")
    print(f"{len(chunks)} chunks created.")
    return chunks


def build_vector_store(chunks: list[Document]) -> None:
    print("Creating embeddings and storing in ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        ids=[c.id for c in chunks],
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTOR_STORE_DIR)
    )
    print("Vector store saved!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB vector store.")
    parser.add_argument("--reset", action="store_true", help="Delete existing vector store before ingesting.")
    parser.add_argument("--skip-failed", action="store_true",
                        help="Ingest the readable PDFs even if some PDFs fail to load.")
    args = parser.parse_args()

    # Fail before touching the existing store if Ollama or the embedding model is unavailable.
    get_embeddings()

    if config.VECTOR_STORE_DIR.exists() and any(config.VECTOR_STORE_DIR.iterdir()) and not args.reset:
        print(
            f"Vector store already exists at '{config.VECTOR_STORE_DIR}'.\n"
            "Use --reset to overwrite it: python ingest.py --reset"
        )
        return

    # Load and chunk before deleting anything, so unreadable PDFs never cost the existing store.
    chunks = chunk_documents(load_pdfs(skip_failed=args.skip_failed))

    if args.reset and config.VECTOR_STORE_DIR.exists():
        print("Deleting existing vector store...")
        shutil.rmtree(config.VECTOR_STORE_DIR)
    build_vector_store(chunks)


if __name__ == "__main__":
    try:
        main()
    except (IngestError, ModelUnavailableError) as e:
        sys.exit(f"Error: {e}")
