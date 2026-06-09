# 📚 Modular RAG Pipeline System

A command-line Python application that lets you drop PDF documents into a folder and ask natural language questions directly against your private data — powered by LangChain, FAISS, Groq API (Llama), and OpenAI embeddings.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3-purple)

---

## 🔍 What Problem Does This Solve?

Standard LLMs don't know your internal or corporate data. When you ask about specific documents, they either hallucinate or say they don't know. This system:

- **Grounds answers in YOUR documents** — the model answers strictly from extracted text
- **Eliminates hallucinations** — a strict prompt template forces context-only answers
- **Saves costs** — FAISS indexes are saved to disk, avoiding redundant embedding API calls

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  PDF Files   │────▶│  PDF Loader  │────▶│  Text Splitter  │
│  (data/)     │     │  (PyPDF)     │     │  (1000/200)     │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────┐     ┌─────────▼────────┐
                    │  FAISS Index │◀────│   Embeddings     │
                    │  (disk save) │     │   (OpenAI)       │
                    └──────┬───────┘     └──────────────────┘
                           │
                    ┌──────▼───────┐     ┌─────────────────┐
                    │  Retriever   │────▶│   QA Chain       │
                    │  (top-k=4)   │     │   (LangChain)    │
                    └──────────────┘     └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  Groq API       │
                                          │  (Llama 3.3)    │
                                          └─────────────────┘
```

---

## 📁 Project Structure

```
Rag/
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
├── data/                    # Drop your PDF files here
│   └── .gitkeep
├── faiss_index/             # Auto-generated vector index (gitignored)
└── src/
    ├── __init__.py
    ├── config.py            # Centralized configuration
    ├── pdf_loader.py        # PDF ingestion with PyPDF
    ├── text_splitter.py     # Chunking with overlap
    ├── embeddings.py        # OpenAI embeddings
    ├── vector_store.py      # FAISS save/load
    ├── retriever.py         # Similarity search
    ├── llm_client.py        # Groq + Llama client
    ├── chain.py             # LangChain QA chain
    └── pipeline.py          # Orchestrator
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Rag-Pipeline.git
cd Rag-Pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)  
> Get an OpenAI API key at [platform.openai.com](https://platform.openai.com)

### 5. Add PDF Documents

Drop your PDF files into the `data/` folder.

### 6. Run the Application

```bash
python main.py
```

---

## 💡 Usage Example

```
============================================================
  📚 Modular RAG Pipeline System
  Ask questions about your PDF documents
============================================================

🚀 Initializing RAG Pipeline
[1/5] Setting up embedding model...
[2/5] Checking for existing FAISS index...
[3/5] Loading PDF documents...
  📄 Loading: company_handbook.pdf
  ✅ Loaded 45 pages from 1 PDF(s)
[4/5] Splitting documents into chunks...
  🔪 Split 45 pages into 128 chunks
[5/5] Building FAISS vector store...
  ✅ FAISS index created successfully
  💾 FAISS index saved to: faiss_index/

✅ Pipeline ready! You can now ask questions.

Type your question below (or 'exit' to quit):

❓ You: What is the company's remote work policy?

💡 Answer: According to the handbook, employees may work remotely
   up to 3 days per week with manager approval...

  📌 Sources:
    - company_handbook.pdf (Page 12)
    - company_handbook.pdf (Page 13)
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Chunk size 1000 + overlap 200** | Prevents splitting sentences mid-way, keeping semantic context intact |
| **FAISS local persistence** | Saves embedding costs — vectors are computed once and reused |
| **Strict prompt template** | Forces the LLM to answer only from context, eliminating hallucinations |
| **Groq + Llama 3.3** | Open-source model with lightning-fast inference via Groq's LPU |
| **Modular architecture** | Each component is independently testable and swappable |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LangChain** — Prompt management & chain orchestration
- **FAISS** — Local vector similarity search
- **Groq API** — Ultra-fast LLM inference
- **Llama 3.3 70B** — Open-source large language model
- **OpenAI Embeddings** — Text vectorization
- **PyPDF** — PDF text extraction
- **python-dotenv** — Environment variable management

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
