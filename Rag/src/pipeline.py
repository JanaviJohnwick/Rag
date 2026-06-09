"""
Pipeline Orchestrator Module.
Ties all modules together into a single, reusable pipeline.
"""

from src.pdf_loader import load_pdfs
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, save_vector_store, load_vector_store
from src.retriever import get_retriever
from src.llm_client import get_llm
from src.chain import build_qa_chain


class RAGPipeline:
    """
    End-to-end RAG Pipeline.

    Orchestrates PDF loading, text splitting, embedding, vector storage,
    retrieval, and question answering.
    """

    def __init__(self):
        self.qa_chain = None
        self._initialize()

    def _initialize(self):
        """Build the full pipeline: load → split → embed → store → chain."""
        print("=" * 60)
        print("🚀 Initializing RAG Pipeline")
        print("=" * 60 + "\n")

        # Step 1: Initialize embeddings
        print("[1/5] Setting up embedding model...")
        embeddings = get_embeddings()

        # Step 2: Try to load existing FAISS index from disk
        print("[2/5] Checking for existing FAISS index...")
        vector_store = load_vector_store(embeddings)

        if vector_store is None:
            # Step 3a: Load PDFs
            print("[3/5] Loading PDF documents...")
            documents = load_pdfs()

            # Step 3b: Split into chunks
            print("[4/5] Splitting documents into chunks...")
            chunks = split_documents(documents)

            # Step 3c: Create and save vector store
            print("[5/5] Building FAISS vector store...")
            vector_store = create_vector_store(chunks, embeddings)
            save_vector_store(vector_store)
        else:
            print("[3/5] ⏩ Skipping PDF loading (using saved index)")
            print("[4/5] ⏩ Skipping text splitting (using saved index)")
            print("[5/5] ⏩ Skipping embedding (using saved index)\n")

        # Step 4: Create retriever and QA chain
        retriever = get_retriever(vector_store)
        llm = get_llm()
        self.qa_chain = build_qa_chain(llm, retriever)

        print("=" * 60)
        print("✅ Pipeline ready! You can now ask questions.")
        print("=" * 60 + "\n")

    def ask(self, question: str) -> dict:
        """
        Ask a question against the loaded documents.

        Args:
            question: The natural language question to ask.

        Returns:
            A dict with 'result' (answer string) and 'source_documents'.
        """
        response = self.qa_chain.invoke({"query": question})
        return response
