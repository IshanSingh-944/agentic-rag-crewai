# Agentic RAG (CrewAI + Ollama gemma:7b)

A retrieval-augmented generation pipeline built as a **multi-agent crew**
rather than a single retrieve-then-generate chain. Three agents each own
one narrow responsibility, running fully locally against Ollama.

## Why "agentic" and not just RAG

A standard RAG pipeline retrieves chunks and generates an answer in one pass,
with no check on whether the retrieved context is actually good. Here, a
dedicated **Grader Agent** sits between retrieval and generation - it can
discard irrelevant chunks or explicitly flag "no relevant context found"
rather than letting the Answer agent hallucinate around bad context.

## Architecture

```
User Query
    |
    v
[Retriever Agent] --uses--> document_retriever tool (FAISS similarity search)
    |
    v
[Grader Agent] -- filters chunks for genuine relevance
    |
    v
[Answer Agent] -- synthesizes final answer from graded context only
    |
    v
Final Answer
```

Agents run in a fixed **sequential** process (`allow_delegation=False` on
all agents) - a deliberate reliability tradeoff for a local 7B model, not
an oversight. See inline comments in `src/agents.py` for the reasoning.

## Stack

- **CrewAI** - multi-agent orchestration
- **Ollama (gemma:7b)** - local LLM inference, no API costs/keys
- **FAISS** - vector store
- **sentence-transformers (all-MiniLM-L6-v2)** - 384-dim embeddings
- **LangChain** - document loaders + text splitting utilities

## Setup

1. Install and start Ollama, and pull the model:
   ```
   ollama pull gemma:7b
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Add your documents (`.pdf`, `.txt`, `.md`) to `data/`.
   A placeholder note is included so you can smoke-test the pipeline first.

4. Build the FAISS index:
   ```
   python -m src.ingest
   ```

5. Run the interactive CLI:
   ```
   python -m src.main
   ```

## Project structure

```
agentic-rag-crewai/
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py     # central tunables (chunk size, top-k, model names)
│   ├── ingest.py      # load -> chunk -> embed -> FAISS index
│   ├── tools.py       # CrewAI retriever tool over the FAISS index
│   ├── agents.py      # Retriever / Grader / Answer agent definitions
│   ├── tasks.py       # task chain wiring agent outputs together
│   ├── crew.py         # assembles agents + tasks into a runnable Crew
│   └── main.py         # interactive CLI entry point
├── data/               # your source documents go here
└── tests/
```

## Known tradeoffs / future improvements

- Full re-index on every ingest run (no incremental upserts).
- Each query rebuilds agents/tasks fresh - no persistent multi-turn memory
  across queries yet.
- Delegation disabled between agents for reliability on a local 7B model;
  could be revisited with a larger/hosted model.
