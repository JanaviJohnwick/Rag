"""
Text Splitter Module.
Splits documents into chunks with tuned size and overlap to preserve
semantic sentence boundaries.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries (paragraphs → sentences → words) to keep semantic meaning
    intact. The overlap ensures no sentence is broken across chunk boundaries.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  🔪 Split {len(documents)} pages into {len(chunks)} chunks")
    print(f"     (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")
    return chunks
