"""
Embeddings Module.
Provides the OpenAI embedding model for vectorizing text chunks.
"""

from langchain_openai import OpenAIEmbeddings
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL


def get_embeddings() -> OpenAIEmbeddings:
    """
    Initialize and return the OpenAI embeddings model.

    Returns:
        An OpenAIEmbeddings instance configured with the specified model.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please add it to your .env file."
        )

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
