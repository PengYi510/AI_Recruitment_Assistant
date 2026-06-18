"""测试向量数据库客户端"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestVectorDBClient:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.config.CHROMA_PERSIST_DIR", tmp_path / "chroma")
        from backend.vector_db.client import VectorDBClient
        return VectorDBClient()

    def test_add_candidate(self, client):
        embedding = np.random.rand(1024).tolist()
        client.add_candidate(1, embedding, {"name": "张三"})
        assert client.get_collection_count() >= 1

    def test_search_similar(self, client):
        # 添加几个候选人
        for i in range(5):
            emb = np.random.rand(1024).tolist()
            client.add_candidate(i+1, emb, {"name": f"候选人_{i}"})
        # 搜索
        query = np.random.rand(1024).tolist()
        results = client.search_similar(query, top_k=3)
        assert len(results) <= 3
        assert all("candidate_id" in r for r in results)

    def test_delete_candidate(self, client):
        emb = np.random.rand(1024).tolist()
        client.add_candidate(100, emb, {"name": "测试"})
        count_before = client.get_collection_count()
        client.delete_candidate(100)
        count_after = client.get_collection_count()
        assert count_after <= count_before

    def test_reset(self, client):
        emb = np.random.rand(1024).tolist()
        client.add_candidate(1, emb, {"name": "test"})
        client.reset()
        assert client.get_collection_count() == 0


class TestVectorDBFallback:
    """Test the in-memory fallback path when ChromaDB is unavailable"""

    @pytest.fixture
    def fallback_client(self):
        """Create a VectorDBClient with _collection=None to test fallback paths"""
        with patch("backend.vector_db.client.CHROMA_AVAILABLE", False):
            from backend.vector_db.client import VectorDBClient
            client = VectorDBClient()
        # Ensure fallback mode
        client._collection = None
        client._fallback_store = []
        return client

    def test_fallback_add_candidate(self, fallback_client):
        embedding = [0.1] * 10
        fallback_client.add_candidate(1, embedding, {"name": "测试"})
        assert fallback_client.get_collection_count() == 1
        assert fallback_client._fallback_store[0]["id"] == "candidate_1"

    def test_fallback_add_candidate_no_metadata(self, fallback_client):
        embedding = [0.2] * 10
        fallback_client.add_candidate(2, embedding)
        assert fallback_client._fallback_store[0]["metadata"] == {}

    def test_fallback_search_similar(self, fallback_client):
        # Add candidates
        fallback_client.add_candidate(1, [1.0, 0.0, 0.0], {"name": "A"})
        fallback_client.add_candidate(2, [0.0, 1.0, 0.0], {"name": "B"})
        fallback_client.add_candidate(3, [0.7, 0.7, 0.0], {"name": "C"})
        # Search for vector close to candidate 1
        results = fallback_client.search_similar([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["candidate_id"] == 1  # Most similar
        assert "distance" in results[0]
        assert "metadata" in results[0]

    def test_fallback_search_empty(self, fallback_client):
        results = fallback_client.search_similar([1.0, 0.0], top_k=5)
        assert results == []

    def test_fallback_delete_candidate(self, fallback_client):
        fallback_client.add_candidate(10, [0.1]*5, {"name": "ten"})
        fallback_client.add_candidate(20, [0.2]*5, {"name": "twenty"})
        assert fallback_client.get_collection_count() == 2
        fallback_client.delete_candidate(10)
        assert fallback_client.get_collection_count() == 1
        assert fallback_client._fallback_store[0]["candidate_id"] == 20

    def test_fallback_reset(self, fallback_client):
        fallback_client.add_candidate(1, [0.1]*5, {})
        fallback_client.add_candidate(2, [0.2]*5, {})
        fallback_client.reset()
        assert fallback_client.get_collection_count() == 0

    def test_fallback_get_collection_count(self, fallback_client):
        assert fallback_client.get_collection_count() == 0
        fallback_client.add_candidate(1, [0.5]*3, {})
        assert fallback_client.get_collection_count() == 1
        fallback_client.add_candidate(2, [0.3]*3, {})
        assert fallback_client.get_collection_count() == 2
