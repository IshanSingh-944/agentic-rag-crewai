# Sample Project Note (Placeholder)

This is a placeholder document so you can smoke-test the ingestion and
retrieval pipeline before dropping in your real resume/notes.

## About this project

This is the Agentic RAG project, built using CrewAI for multi-agent
orchestration and a locally running Ollama gemma:7b model for inference.
It uses FAISS as the vector store and sentence-transformers
(all-MiniLM-L6-v2) for generating embeddings.

## How to test

Replace this file with your actual resume (PDF) and project notes (.md/.txt),
then re-run:

    python -m src.ingest

Then start the interactive CLI:

    python -m src.main

Try asking: "What is this project about?" - it should retrieve this
paragraph and answer using it.
