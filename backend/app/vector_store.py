import hashlib
import logging
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDER_INSTANCE = None


def get_sentence_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Singleton loader for SentenceTransformer to avoid reloading weights across requests."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            _EMBEDDER_INSTANCE = SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning(f"SentenceTransformer load fallback triggered: {exc}")
            _EMBEDDER_INSTANCE = False
    return _EMBEDDER_INSTANCE


class FAISSVectorMemory:
    """
    In-memory + persistable FAISS Vector Index powered by SentenceTransformers (all-MiniLM-L6-v2).
    Stores crawled web chunks, search snippets, and evidence with metadata for semantic RAG retrieval.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict[str, Any]] = []
        self._seen_hashes = set()

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embedder = get_sentence_embedder(self.model_name)
        if embedder:
            embeddings = embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.astype("float32")

        # Deterministic hashed trigram projection fallback if offline/weights unavailable
        vecs = np.zeros((len(texts), self.dimension), dtype="float32")
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vecs[i, h % self.dimension] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs

    @staticmethod
    def chunk_text(text: str, max_chars: int = 700, overlap: int = 100) -> List[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        if len(cleaned) <= max_chars:
            return [cleaned]
        chunks = []
        start = 0
        while start < len(cleaned):
            end = min(start + max_chars, len(cleaned))
            chunks.append(cleaned[start:end])
            if end == len(cleaned):
                break
            start = max(0, end - overlap)
        return chunks

    def add_documents(self, items: List[Dict[str, str]]) -> int:
        """
        Chunks, embeds, and indexes a list of documents into FAISS.
        Each item has keys: {'title': ..., 'url': ..., 'content': ..., 'source_tool': ...}
        Returns the number of new chunks added.
        """
        new_chunks_meta: List[Dict[str, Any]] = []
        new_chunk_texts: List[str] = []

        for item in items:
            content = item.get("content", "").strip()
            title = item.get("title", "Untitled Source").strip()
            url = item.get("url", "local://memory").strip()
            source_tool = item.get("source_tool", "unknown")
            if not content:
                continue

            chunks = self.chunk_text(content)
            for idx, chunk in enumerate(chunks):
                h = hashlib.sha256(f"{url}::{chunk}".encode("utf-8")).hexdigest()
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
                combined_text = f"{title}: {chunk}"
                new_chunk_texts.append(combined_text)
                new_chunks_meta.append(
                    {
                        "chunk_id": len(self.documents) + len(new_chunks_meta),
                        "title": title,
                        "url": url,
                        "chunk_index": idx,
                        "content": chunk,
                        "source_tool": source_tool,
                    }
                )

        if not new_chunk_texts:
            return 0

        vectors = self._embed_texts(new_chunk_texts)
        self.index.add(vectors)
        self.documents.extend(new_chunks_meta)
        return len(new_chunk_texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs cosine-similarity search over the FAISS index."""
        if self.index.ntotal == 0 or not query.strip():
            return []

        k = min(top_k, self.index.ntotal)
        query_vec = self._embed_texts([query])
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = dict(self.documents[idx])
            doc["similarity_score"] = round(float(score), 4)
            results.append(doc)
        return results

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
