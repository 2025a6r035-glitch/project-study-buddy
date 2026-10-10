"""Notes ingestion + retrieval: read files -> chunk -> embed -> FAISS search."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from .llm import LocalEmbedder

try:  # faiss is optional at runtime; numpy brute-force is the fallback
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


@dataclass
class Chunk:
    text: str
    source: str
    idx: int


def read_file(name: str, data: bytes) -> str:
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext in ("txt", "md", "markdown", ""):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type '.{ext}'. Use PDF, TXT or MD.")


def chunk_text(text: str, size: int = 250, overlap: int = 40) -> list[str]:
    """Split into overlapping windows of ~`size` words."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if start > 0 and len(window) <= overlap + 5:
            break  # tail is already covered by the previous window's overlap
        chunks.append(" ".join(window))
    return chunks


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray((m / norms).astype("float32"))


class NoteIndex:
    """Vector index over the student's notes (cosine similarity)."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self._faiss = None
        self.warning = ""

    def __len__(self) -> int:
        return len(self.chunks)

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        return _normalize(self.embedder.embed(texts, task))

    def _rebuild_faiss(self) -> None:
        if faiss is not None and self.vectors is not None:
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)

    def add(self, source: str, text: str) -> int:
        pieces = chunk_text(text)
        if not pieces:
            return 0
        new = [Chunk(p, source, len(self.chunks) + i) for i, p in enumerate(pieces)]
        try:
            vecs = self._embed([c.text for c in new], "RETRIEVAL_DOCUMENT")
        except Exception as exc:
            if isinstance(self.embedder, LocalEmbedder):
                raise
            # Gemini embeddings failed: switch the WHOLE index to local embeddings.
            self.warning = f"Gemini embeddings unavailable ({exc}); using offline keyword embeddings instead."
            self.embedder = LocalEmbedder()
            self.vectors = self._embed([c.text for c in self.chunks], "RETRIEVAL_DOCUMENT") if self.chunks else None
            vecs = self._embed([c.text for c in new], "RETRIEVAL_DOCUMENT")
        self.chunks.extend(new)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self._rebuild_faiss()
        return len(new)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks or self.vectors is None:
            return []
        q = self._embed([query], "RETRIEVAL_QUERY")
        k = min(k, len(self.chunks))
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = self.vectors @ q[0]
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order]

    def sample_text(self, max_chars: int = 6000) -> str:
        """Evenly spaced excerpts across all notes (used to auto-detect topics)."""
        if not self.chunks:
            return ""
        per = 700
        n = max(1, max_chars // per)
        step = max(1, len(self.chunks) // n)
        picked = self.chunks[::step][:n]
        return "\n\n".join(c.text[:per] for c in picked)

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.source] = out.get(c.source, 0) + 1
        return out
