"""
PDF Document Loader Module.
Loads all PDF files from the configured data folder using PyPDF.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from src.config import PDF_FOLDER


def load_pdfs(folder_path: str = PDF_FOLDER) -> list:
    """
    Load all PDF files from the specified folder.

    Args:
        folder_path: Path to the folder containing PDF files.

    Returns:
        A list of LangChain Document objects with page content and metadata.

    Raises:
        FileNotFoundError: If the folder doesn't exist.
        ValueError: If no PDF files are found in the folder.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"PDF folder not found: {folder_path}")

    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in '{folder_path}'. "
            "Please add PDF documents to the data/ folder."
        )

    documents = []
    for pdf_path in pdf_files:
        print(f"  📄 Loading: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    print(f"  ✅ Loaded {len(documents)} pages from {len(pdf_files)} PDF(s)\n")
    return documents
