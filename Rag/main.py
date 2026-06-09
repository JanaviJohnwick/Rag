"""
Modular RAG Pipeline — CLI Entry Point.
Drop PDFs into the data/ folder and ask natural language questions
against your private documents.
"""

import sys
from src.pipeline import RAGPipeline


def main():
    """Interactive REPL for querying documents."""
    print("\n" + "=" * 60)
    print("  📚 Modular RAG Pipeline System")
    print("  Ask questions about your PDF documents")
    print("=" * 60 + "\n")

    try:
        pipeline = RAGPipeline()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n❌ Setup Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error during initialization: {e}")
        sys.exit(1)

    # ── Interactive question loop ───────────────────────────────────────
    print("Type your question below (or 'exit' to quit):\n")

    while True:
        try:
            question = input("❓ You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("\n👋 Goodbye!")
            break

        try:
            response = pipeline.ask(question)
            answer = response.get("result", "No answer generated.")
            sources = response.get("source_documents", [])

            print(f"\n💡 Answer: {answer}\n")

            # Show source references
            if sources:
                print("  📌 Sources:")
                seen = set()
                for doc in sources:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    ref = f"    - {source} (Page {page})"
                    if ref not in seen:
                        seen.add(ref)
                        print(ref)
                print()

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
