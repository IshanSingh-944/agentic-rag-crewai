"""
Retriever tool exposed to CrewAI agents.

We wrap the FAISS similarity search in a crewai_tools BaseTool so the
Retriever Agent can call it like any other tool during task execution.
"""

import os
from typing import Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import INDEX_DIR, EMBEDDING_MODEL_NAME, TOP_K


class RetrieverInput(BaseModel):
    query: str = Field(..., description="The search query to find relevant document chunks for.")


class DocumentRetrieverTool(BaseTool):
    name: str = "document_retriever"
    description: str = (
        "Searches the personal document knowledge base (resume, project notes) "
        "and returns the most relevant chunks for a given query. "
        "Use this whenever you need factual context before answering."
    )
    args_schema: Type[BaseModel] = RetrieverInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.path.isdir(INDEX_DIR):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_DIR}. "
                "Run `python -m src.ingest` first to build it."
            )
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # allow_dangerous_deserialization: safe here since we only ever
        # load an index we ourselves created in ingest.py.
        self._vectorstore = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )

    def _run(self, query: str) -> str:
        results = self._vectorstore.similarity_search(query, k=TOP_K)
        if not results:
            return "No relevant context found in the knowledge base."

        formatted = []
        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[Chunk {i} | source: {source}]\n{doc.page_content}")

        return "\n\n".join(formatted)
