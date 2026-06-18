"""测试配置模块"""
import pytest
from pathlib import Path


def test_config_imports():
    from backend.config import (
        LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
        DB_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
        HARNESS_MAX_ITERATIONS, COMPLEXITY_THRESHOLD,
        RAG_TOP_K, FINAL_TOP_K, SHAP_FEATURE_NAMES,
    )
    assert LLM_MODEL is not None
    assert len(SHAP_FEATURE_NAMES) == 12


def test_config_paths():
    from backend.config import SHAP_DIR, CHROMA_PERSIST_DIR
    assert isinstance(SHAP_DIR, Path)
    assert isinstance(CHROMA_PERSIST_DIR, Path)


def test_catboost_config():
    from backend.config import CATBOOST_ITERATIONS, CATBOOST_LEARNING_RATE, CATBOOST_DEPTH
    assert CATBOOST_ITERATIONS >= 100
    assert 0 < CATBOOST_LEARNING_RATE < 1
    assert CATBOOST_DEPTH >= 4
