"""
Vector Store Module.
Manages FAISS index creation, saving, and loading from disk
to avoid redundant embedding API costs.
"""

import os
from langchain_community.vectorstores import FAISS
from src.config import FAISS_INDEX_PATH


def create_vector_store(chunks: list, embeddings) -> FAISS:
    """
    Create a FAISS vector store from document chunks.

    Args:
        chunks: List of chunked Document objects.
        embeddings: Embedding model instance.

    Returns:
        A FAISS vector store instance.
    """
    print("  🧠 Generating embeddings and building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("  ✅ FAISS index created successfully\n")
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = FAISS_INDEX_PATH) -> None:
    """
    Save FAISS index to disk for reuse.

    Args:
        vector_store: The FAISS vector store to save.
        path: Directory path to save the index to.
    """
    vector_store.save_local(path)
    print(f"  💾 FAISS index saved to: {path}\n")


def load_vector_store(embeddings, path: str = FAISS_INDEX_PATH) -> FAISS | None:
    """
    Load a previously saved FAISS index from disk.

    Args:
        embeddings: Embedding model instance (needed for deserialization).
        path: Directory path where the index is stored.

    Returns:
        A FAISS vector store instance, or None if no saved index exists.
    """
    index_file = os.path.join(path, "index.faiss")
    if os.path.exists(index_file):
        print(f"  📂 Loading existing FAISS index from: {path}")
        vector_store = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )
        print("  ✅ FAISS index loaded successfully\n")
        return vector_store

    return None
