"""
Ingestion pipeline: load documents from data/, split into chunks,
embed them, and persist a FAISS index to disk.

Run standalone: python -m src.ingest
"""

import os
import sys

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import (
    DATA_DIR,
    INDEX_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents(data_dir: str = DATA_DIR):
    """Load PDFs, .txt, and .md files from data_dir."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs = []

    pdf_loader = DirectoryLoader(
        data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    docs.extend(pdf_loader.load())

    # TextLoader handles both .txt and .md fine since both are plain text
    for pattern in ("**/*.txt", "**/*.md"):
        text_loader = DirectoryLoader(
            data_dir, glob=pattern, loader_cls=TextLoader
        )
        docs.extend(text_loader.load())

    if not docs:
        raise ValueError(
            f"No .pdf, .txt, or .md files found in {data_dir}. "
            "Add some documents before ingesting."
        )

    return docs


def chunk_documents(docs, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_faiss_index(chunks, index_dir: str = INDEX_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    return vectorstore


def main():
    print(f"Loading documents from {DATA_DIR} ...")
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s).")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks)} chunk(s).")

    print(f"Embedding with {EMBEDDING_MODEL_NAME} and building FAISS index ...")
    build_faiss_index(chunks)
    print(f"Index saved to {INDEX_DIR}")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
