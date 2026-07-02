"""The conversational RAG pipeline, composed from LCEL primitives (LangChain 1.x).

Two-stage design:
1. If there is chat history, the user's question is first rewritten into a
   standalone query ("and after that?" -> "what happens after 30 days?") so
   retrieval works mid-conversation.
2. The answer is generated strictly from the retrieved chunks, streamed token
   by token. Retrieval happens before generation, so the source documents are
   available up front for citations.
"""

from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama

from src.config import settings

# Rewrites follow-up questions into standalone queries so retrieval works mid-conversation.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the question "
            "as a standalone question that can be understood without the history. "
            "Do NOT answer it — only rewrite it, or return it unchanged if already standalone.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Grounded answering: the model may only use retrieved context, cutting hallucination.
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant answering questions about an uploaded PDF document. "
            "Answer using ONLY the context below. If the answer is not in the context, "
            "say you don't know — do not make anything up. Be concise.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,  # low temperature: factual answers over creativity
    )


def format_context(docs: list[Document]) -> str:
    """Retrieved chunks as a prompt-ready block, page-tagged so the model can cite."""
    return "\n\n".join(
        f"[page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}" for doc in docs
    )


def extract_source_pages(docs: list[Document]) -> list[int]:
    """Unique, sorted 1-based page numbers of the chunks an answer was drawn from."""
    return sorted({doc.metadata.get("page", 0) + 1 for doc in docs})


class ConversationalRag:
    """History-aware retrieval + grounded, streamed answering over one document."""

    def __init__(self, retriever: BaseRetriever, llm: ChatOllama | None = None):
        self._retriever = retriever
        llm = llm or get_llm()
        self._condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
        self._answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    def retrieve(self, question: str, chat_history: list[BaseMessage]) -> list[Document]:
        """Fetch the top-k chunks, condensing the question first if mid-conversation."""
        if chat_history:
            question = self._condense_chain.invoke(
                {"input": question, "chat_history": chat_history}
            )
        return self._retriever.invoke(question)

    def stream_answer(
        self, question: str, chat_history: list[BaseMessage]
    ) -> tuple[Iterator[str], list[Document]]:
        """Return (token stream, source documents) for a question.

        The documents are resolved eagerly — before any token is generated — so
        the caller can render citations as soon as the answer finishes streaming.
        """
        docs = self.retrieve(question, chat_history)
        tokens = self._answer_chain.stream(
            {
                "input": question,
                "chat_history": chat_history,
                "context": format_context(docs),
            }
        )
        return tokens, docs


def build_rag_chain(retriever: BaseRetriever) -> ConversationalRag:
    return ConversationalRag(retriever)
