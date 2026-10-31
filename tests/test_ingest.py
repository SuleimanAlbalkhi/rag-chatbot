import pytest

import config
import ingest
from tests.conftest import BagOfWordsEmbeddings
from tests.pdf_factory import make_pdf


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(ingest, "get_embeddings", BagOfWordsEmbeddings)


def docs_dir():
    return config.DOCUMENTS_DIR


def test_pages_are_one_based_with_page_labels():
    pdf = make_pdf(
        docs_dir() / "a.pdf",
        ["front", "intro", "body one", "body two"],
        page_labels="<< /Nums [0 << /S /r >> 2 << /S /D >>] >>",
    )
    pages = ingest.load_pdf(pdf)
    assert [p.metadata["page"] for p in pages] == [1, 2, 3, 4]
    assert [p.metadata["page_label"] for p in pages] == ["i", "ii", "1", "2"]
    assert pages[2].page_content.strip() == "body one"
    assert pages[0].metadata["source"] == "a.pdf"


def test_chunk_ids_are_deterministic_and_readable(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SIZE", 100)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0)
    make_pdf(docs_dir() / "gdpr.pdf", ["short page", "word " * 60])
    first = ingest.chunk_documents(ingest.load_pdfs())
    second = ingest.chunk_documents(ingest.load_pdfs())
    ids = [c.id for c in first]
    assert ids == [c.id for c in second]
    assert ids[0] == "gdpr.pdf:p1:c0"
    assert ids[1:] == [f"gdpr.pdf:p2:c{i}" for i in range(len(ids) - 1)]
    assert len(set(ids)) == len(ids)


def test_no_pdfs_is_an_error():
    with pytest.raises(ingest.IngestError, match="No PDFs"):
        ingest.load_pdfs()


def test_unreadable_and_textless_pdfs_abort_by_default():
    make_pdf(docs_dir() / "good.pdf", ["Article 1 text"])
    (docs_dir() / "corrupt.pdf").write_bytes(b"not a pdf")
    make_pdf(docs_dir() / "scanned.pdf", ["", ""])
    with pytest.raises(ingest.IngestError) as e:
        ingest.load_pdfs()
    assert "corrupt.pdf" in str(e.value) and "scanned.pdf" in str(e.value)


def test_skip_failed_keeps_readable_pdfs():
    make_pdf(docs_dir() / "good.pdf", ["Article 1 text"])
    (docs_dir() / "corrupt.pdf").write_bytes(b"not a pdf")
    docs = ingest.load_pdfs(skip_failed=True)
    assert {d.metadata["source"] for d in docs} == {"good.pdf"}


def test_skip_failed_still_fails_when_nothing_is_readable():
    (docs_dir() / "corrupt.pdf").write_bytes(b"not a pdf")
    with pytest.raises(ingest.IngestError):
        ingest.load_pdfs(skip_failed=True)


def run_main(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["ingest.py", *args])
    ingest.main()


def stored_ids():
    import models

    return sorted(models.get_vector_store(BagOfWordsEmbeddings()).get()["ids"])


def test_main_builds_store_with_chunk_ids(monkeypatch):
    make_pdf(docs_dir() / "ehds.pdf", ["Cover", "Article 1 subject matter"])
    run_main(monkeypatch)
    assert stored_ids() == ["ehds.pdf:p1:c0", "ehds.pdf:p2:c0"]


def test_reset_keeps_existing_store_when_pdfs_fail(monkeypatch):
    make_pdf(docs_dir() / "ehds.pdf", ["Cover", "Article 1 subject matter"])
    run_main(monkeypatch)
    (docs_dir() / "corrupt.pdf").write_bytes(b"not a pdf")
    with pytest.raises(ingest.IngestError):
        run_main(monkeypatch, "--reset")
    assert stored_ids() == ["ehds.pdf:p1:c0", "ehds.pdf:p2:c0"]
