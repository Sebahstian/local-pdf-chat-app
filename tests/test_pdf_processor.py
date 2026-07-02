"""Unit tests for PDF loading and chunking — pure logic, no Ollama required."""

from src.chain import extract_source_pages
from src.config import settings
from src.pdf_processor import load_pdf, split_documents


def test_load_pdf_returns_one_document_per_page(sample_pdf_bytes):
    docs = load_pdf(sample_pdf_bytes)
    assert len(docs) == 3
    assert all(doc.page_content.strip() for doc in docs)


def test_pages_carry_page_metadata(sample_pdf_bytes):
    docs = load_pdf(sample_pdf_bytes)
    assert [doc.metadata["page"] for doc in docs] == [0, 1, 2]


def test_split_documents_produces_bounded_chunks(sample_pdf_bytes):
    chunks = split_documents(load_pdf(sample_pdf_bytes))
    assert len(chunks) >= 3  # long pages must split into multiple chunks
    assert all(len(c.page_content) <= settings.chunk_size for c in chunks)


def test_chunks_inherit_page_metadata(sample_pdf_bytes):
    chunks = split_documents(load_pdf(sample_pdf_bytes))
    assert all("page" in c.metadata for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {0, 1, 2}


def test_extract_source_pages_is_unique_sorted_one_based(sample_pdf_bytes):
    chunks = split_documents(load_pdf(sample_pdf_bytes))
    pages = extract_source_pages(chunks)
    assert pages == [1, 2, 3]  # 1-based, deduplicated, sorted
