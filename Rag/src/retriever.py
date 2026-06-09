"""
Retriever Module.
Wraps the FAISS vector store as a LangChain retriever for similarity search.
"""

from langchain_community.vectorstores import FAISS
from src.config import RETRIEVER_TOP_K


def get_retriever(vector_store: FAISS):
    """
    Create a retriever from the FAISS vector store.

    Args:
        vector_store: A FAISS vector store instance.

    Returns:
        A LangChain retriever configured to return top-k relevant chunks.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )
