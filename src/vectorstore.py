"""Embedding and vector search. Everything runs locally via Ollama.

The index is langchain-core's InMemoryVectorStore: for a single uploaded PDF the
corpus is small (hundreds of chunks), so exact cosine search is both correct and
dependency-free — no vector DB service, no native FAISS wheel. The store is
rebuilt per upload and lives for the session. Swapping in Chroma/FAISS for
persistence would only touch this file.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from src.config import settings


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )


def build_retriever(chunks: list[Document]) -> VectorStoreRetriever:
    """Embed the chunks into an in-memory index and return a top-k retriever."""
    vectorstore = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
