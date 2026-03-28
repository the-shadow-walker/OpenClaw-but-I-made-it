"""
RAG Tool — ChromaDB wrapper for physics (and future) reference databases.

Fixes the embedding-model mismatch in the old knowledge_retriever.py:
  • Old code queried with all-MiniLM-L6-v2  (wrong)
  • DB was ingested  with BAAI/bge-large-en-v1.5 (correct)
  This file always uses BGE-Large to match the stored embeddings.

Usage:
    from rag_tool import rag_search
    chunks = rag_search("Tsiolkovsky rocket equation", domain="physics", n=5)
    # Returns newline-separated chunks, or "" on failure.
"""

import os
import warnings
from typing import Optional

# ── Database / collection registry ──────────────────────────────────────────
DB_PATHS = {
    "physics": "/mnt/storage/NAS/Jarvis/RAG/phys/physics_db",
    # future: "chemistry": "/mnt/storage/NAS/Jarvis/RAG/chem/chem_db",
    # future: "electrical": "/mnt/storage/NAS/Jarvis/RAG/elec/elec_db",
}
COLLECTION_NAMES = {
    "physics": "physics_reference",
}
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"   # MUST match ingest — do not change

# ── Module-level cache: one client per domain ────────────────────────────────
_clients: dict = {}       # domain → chromadb.PersistentClient
_collections: dict = {}   # domain → chromadb.Collection
_ef: object = None         # shared embedding function


def _get_embedding_function():
    """Lazy-load the BGE-Large embedding function (cached after first call)."""
    global _ef
    if _ef is not None:
        return _ef
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        return _ef
    except Exception as e:
        print(f"⚠️  RAG: could not load embedding function ({e})")
        return None


def _get_collection(domain: str):
    """Return (cached) ChromaDB collection for the given domain, or None."""
    if domain in _collections:
        return _collections[domain]

    db_path = DB_PATHS.get(domain)
    coll_name = COLLECTION_NAMES.get(domain)
    if not db_path or not coll_name:
        print(f"⚠️  RAG: unknown domain '{domain}'")
        return None
    if not os.path.isdir(db_path):
        print(f"⚠️  RAG: DB not found at {db_path}")
        return None

    ef = _get_embedding_function()
    if ef is None:
        return None

    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        _clients[domain] = client
        coll = client.get_collection(name=coll_name, embedding_function=ef)
        _collections[domain] = coll
        return coll
    except Exception as e:
        print(f"⚠️  RAG: failed to open collection '{coll_name}' in {db_path}: {e}")
        return None


def rag_search(query: str, domain: str = "physics", n: int = 5) -> str:
    """
    Retrieve the top-N most relevant reference chunks from the RAG database.

    Args:
        query:  Natural-language or keyword query.
        domain: Database domain key (default "physics").
        n:      Number of chunks to return.

    Returns:
        String with chunks separated by "\\n---\\n", or "" on any failure.
        Caller should treat "" as "no RAG context available" and continue.
    """
    try:
        coll = _get_collection(domain)
        if coll is None:
            return ""

        results = coll.query(
            query_texts=[query],
            n_results=min(n, coll.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        chunks = []
        for doc, meta, dist in zip(docs, metas, dists):
            source = meta.get("source", "unknown") if meta else "unknown"
            page = meta.get("page", "") if meta else ""
            header = f"[{source}" + (f", p.{page}" if page else "") + f", dist={dist:.3f}]"
            chunks.append(f"{header}\n{doc}")

        return "\n---\n".join(chunks)

    except Exception as e:
        print(f"⚠️  RAG search error: {e}")
        return ""
