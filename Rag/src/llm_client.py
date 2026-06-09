"""
LLM Client Module.
Initializes the Groq API client with Llama model for fast inference.
"""

from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


def get_llm() -> ChatGroq:
    """
    Initialize and return the Groq LLM client.

    Uses Llama models hosted on Groq for lightning-fast inference times.

    Returns:
        A ChatGroq LLM instance.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Please add it to your .env file."
        )

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
