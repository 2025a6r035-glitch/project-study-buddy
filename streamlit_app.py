import os

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---------- Settings (change model names here if Google retires them) ----------
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks
TOP_K = 4              # how many chunks to retrieve per question
EMBED_BATCH = 50       # chunks sent to the embedding API per request

st.set_page_config(page_title="Study Buddy", page_icon="📚", layout="wide")


# ---------- API key ----------
def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


API_KEY = get_api_key()
if not API_KEY:
    st.error(
        "No API key found. Add `GEMINI_API_KEY` in Manage app → Settings → Secrets "
        "(on Streamlit Cloud) or in a local `.env` / environment variable."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)


# ---------- Helpers ----------
def read_pdfs(files):
    """Return a list of (filename, text) for each uploaded PDF."""
    docs = []
    for f in files:
        reader = PdfReader(f)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        docs.append((f.name, text))
    return docs


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    text = " ".join(text.split())
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def embed(texts, task_type):
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in resp.embeddings)
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_index(files):
    chunks, sources = [], []
    for name, text in read_pdfs(files):
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(name)
    if not chunks:
        return None
    vectors = embed(chunks, "RETRIEVAL_DOCUMENT")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return {"index": index, "chunks": chunks, "sources": sources}


def ask(question, store):
    q_vec = embed([question], "RETRIEVAL_QUERY")
    _, ids = store["index"].search(q_vec, min(TOP_K, len(store["chunks"])))
    context = "\n\n---\n\n".join(
        f"[{store['sources'][i]}]\n{store['chunks'][i]}" for i in ids[0]
    )
    prompt = (
        "You are a friendly study buddy. Answer the student's question using the "
        "notes below. If the notes do not contain the answer, say so, then give a "
        "short general answer and make clear it is not from their notes.\n\n"
        f"NOTES:\n{context}\n\nQUESTION: {question}"
    )
    resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return resp.text, sorted({store["sources"][i] for i in ids[0]})


def generate_quiz(store, n=5):
    sample = "\n\n".join(store["chunks"][:12])
    prompt = (
        f"Write {n} multiple-choice practice questions based on these notes. "
        "Give 4 options (A-D) for each, then list the answers with a one-line "
        f"explanation at the end.\n\nNOTES:\n{sample}"
    )
    return client.models.generate_content(model=CHAT_MODEL, contents=prompt).text


# ---------- UI ----------
st.title("📚 Study Buddy")
st.caption("Upload your notes as PDFs, then ask questions or get a practice quiz.")

if "store" not in st.session_state:
    st.session_state.store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Your notes")
    uploads = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Process notes", type="primary", disabled=not uploads):
        with st.spinner("Reading and indexing your notes..."):
            try:
                store = build_index(uploads)
                if store is None:
                    st.warning("Couldn't find any text in those PDFs (scanned images aren't supported).")
                else:
                    st.session_state.store = store
                    st.session_state.messages = []
                    st.success(f"Indexed {len(store['chunks'])} chunks.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if st.session_state.store and st.button("Make a practice quiz"):
        with st.spinner("Writing quiz..."):
            try:
                st.session_state.messages.append(
                    {"role": "assistant", "content": generate_quiz(st.session_state.store)}
                )
            except Exception as e:
                st.error(f"Quiz failed: {e}")

if not st.session_state.store:
    st.info("👈 Upload your PDFs and click **Process notes** to begin.")
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if question := st.chat_input("Ask something about your notes..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, srcs = ask(question, st.session_state.store)
                    reply = f"{answer}\n\n*Sources: {', '.join(srcs)}*"
                except Exception as e:
                    reply = f"Sorry, something went wrong: {e}"
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
