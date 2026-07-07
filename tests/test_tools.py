"""
Tests for tools.py.

Note: we only test the RetrieverInput schema here. We deliberately do NOT
instantiate DocumentRetrieverTool in unit tests, since its __init__ builds
a real FAISS index and downloads the embedding model - that's an
integration-test concern, not something we want blocking a fast test suite.
"""

import pytest
from pydantic import ValidationError

from src.tools import RetrieverInput


def test_retriever_input_accepts_valid_query():
    data = RetrieverInput(query="What projects has Ishan worked on?")
    assert data.query == "What projects has Ishan worked on?"


def test_retriever_input_requires_query_field():
    with pytest.raises(ValidationError):
        RetrieverInput()
