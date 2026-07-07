"""
Tests for ingest.py.

Note: we deliberately avoid testing build_faiss_index() here since it
requires downloading the embedding model - that's an integration-test
concern (see tests/README note below), not a fast unit test.
"""

import pytest
from langchain_core.documents import Document

from src.ingest import load_documents, chunk_documents
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def test_load_documents_missing_dir_raises(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_documents(str(missing_dir))


def test_load_documents_empty_dir_raises(tmp_path):
    # Directory exists but has no .pdf/.txt/.md files
    with pytest.raises(ValueError):
        load_documents(str(tmp_path))


def test_load_documents_finds_txt_file(tmp_path):
    sample = tmp_path / "note.txt"
    sample.write_text("Hello world, this is a test note.")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content


def test_load_documents_finds_md_file(tmp_path):
    sample = tmp_path / "note.md"
    sample.write_text("# Heading\n\nSome markdown content.")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert "markdown content" in docs[0].page_content


def test_chunk_documents_respects_chunk_size():
    long_text = "word " * 500  # ~2500 chars, well beyond CHUNK_SIZE
    docs = [Document(page_content=long_text, metadata={"source": "fake.txt"})]
    chunks = chunk_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        # allow a little slack since splitter tries not to break mid-word
        assert len(chunk.page_content) <= CHUNK_SIZE + CHUNK_OVERLAP


def test_chunk_documents_preserves_metadata():
    docs = [Document(page_content="short text", metadata={"source": "fake.md"})]
    chunks = chunk_documents(docs)
    assert chunks[0].metadata["source"] == "fake.md"
