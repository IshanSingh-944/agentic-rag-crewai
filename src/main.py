"""
Interactive CLI entry point.

Run with: python -m src.main
Type a question, get an answer from the Retriever -> Grader -> Answer crew.
Type 'exit' or 'quit' to stop.
"""

import os
import sys

from src.config import INDEX_DIR
from src.crew import run_query


def check_index_exists():
    if not os.path.isdir(INDEX_DIR):
        print(
            f"No FAISS index found at {INDEX_DIR}.\n"
            "Run `python -m src.ingest` first to build it from your documents in data/."
        )
        sys.exit(1)


def main():
    check_index_exists()

    print("Agentic RAG (CrewAI + Ollama gemma:7b)")
    print("Ask a question about your documents. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        print("\n--- Crew is working (this may take a while on a 7B local model) ---\n")
        answer = run_query(query)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
