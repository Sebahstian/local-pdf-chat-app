"""PDF loading and chunking. Page metadata is preserved so answers can cite pages."""

import io

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import settings


def load_pdf(file_bytes: bytes) -> list[Document]:
    """Parse a PDF (as raw bytes) into one Document per page.

    Each Document carries {"page": n} metadata (0-indexed) used for citations.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    return [
        Document(page_content=page.extract_text() or "", metadata={"page": i})
        for i, page in enumerate(reader.pages)
    ]


def split_documents(docs: list[Document]) -> list[Document]:
    """Split page documents into overlapping chunks sized for retrieval.

    RecursiveCharacterTextSplitter keeps paragraphs/sentences intact where possible,
    and each chunk inherits its source page's metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)
