"""
Sanity checks on config values. Cheap to run, catches silly mistakes
like overlap >= chunk_size before they cause confusing runtime errors.
"""

from src import config


def test_chunk_overlap_smaller_than_chunk_size():
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_top_k_is_positive():
    assert config.TOP_K > 0


def test_ollama_model_name_set():
    assert config.OLLAMA_MODEL == "gemma:7b"


def test_paths_are_absolute():
    assert config.DATA_DIR.startswith("/") or ":" in config.DATA_DIR  # posix or windows
    assert config.INDEX_DIR.startswith("/") or ":" in config.INDEX_DIR
