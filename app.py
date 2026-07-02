"""Chat with any PDF — fully local RAG (Streamlit + LangChain + Ollama).

Run with: streamlit run app.py
"""

import hashlib

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.chain import build_rag_chain, extract_source_pages
from src.config import settings
from src.pdf_processor import load_pdf, split_documents
from src.vectorstore import build_retriever

st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="centered")


def index_pdf(file_bytes: bytes, file_name: str) -> None:
    """Parse, chunk, embed the PDF and build the RAG chain — once per file."""
    with st.status(f"Indexing **{file_name}** …", expanded=True) as status:
        st.write("Parsing PDF…")
        pages = load_pdf(file_bytes)
        st.write(f"Loaded {len(pages)} pages. Chunking…")
        chunks = split_documents(pages)
        st.write(f"Embedding {len(chunks)} chunks locally ({settings.embed_model})…")
        retriever = build_retriever(chunks)
        st.session_state.chain = build_rag_chain(retriever)
        st.session_state.messages = []
        status.update(
            label=f"Ready — {len(pages)} pages, {len(chunks)} chunks indexed.",
            state="complete",
            expanded=False,
        )


def stream_answer(question: str):
    """Retrieve sources, then stream the answer tokens into the UI."""
    tokens, context_docs = st.session_state.chain.stream_answer(
        question, st.session_state.messages
    )
    answer = st.write_stream(tokens)
    return answer, context_docs


# ── Sidebar: upload ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Chat")
    st.caption(
        f"100% local — `{settings.llm_model}` + `{settings.embed_model}` via Ollama. "
        "Your document never leaves this machine."
    )
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        # Hash guards against re-embedding on every Streamlit rerun.
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get("file_hash") != file_hash:
            index_pdf(file_bytes, uploaded.name)
            st.session_state.file_hash = file_hash

    if st.session_state.get("chain") and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main: chat ───────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to start chatting with it.")
    st.stop()

# Replay history (messages are LangChain Message objects; sources ride in .metadata)
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)
        if pages := msg.additional_kwargs.get("source_pages"):
            st.caption("📄 Sources: " + ", ".join(f"p.{p}" for p in pages))

if question := st.chat_input("Ask something about the PDF…"):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            answer, context_docs = stream_answer(question)
        except Exception as exc:  # most commonly: Ollama not running
            st.error(
                f"Could not reach the model — is Ollama running at "
                f"`{settings.ollama_base_url}`?\n\n`{exc}`"
            )
            st.stop()
        source_pages = extract_source_pages(context_docs)
        if source_pages:
            st.caption("📄 Sources: " + ", ".join(f"p.{p}" for p in source_pages))

    st.session_state.messages.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=answer, additional_kwargs={"source_pages": source_pages}),
        ]
    )
