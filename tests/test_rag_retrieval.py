"""测试RAG检索"""
import pytest
import numpy as np
from unittest.mock import patch


class TestRAGRetrieval:
    @pytest.mark.asyncio
    async def test_bm25_search(self):
        from backend.skills.rag_retrieval_skill import RAGRetrievalSkill
        with patch("backend.skills.rag_retrieval_skill.hr_db") as mock_db, \
             patch("backend.skills.rag_retrieval_skill.vector_db") as mock_vdb, \
             patch("backend.skills.rag_retrieval_skill.multimodal_fusion") as mock_fusion:
            mock_db.search_candidates.return_value = [
                {"id": 1, "name": "张三", "highest_education": "硕士",
                 "education_history": [{"school": "清华大学", "degree": "硕士", "major": "CS"}]},
                {"id": 2, "name": "李四", "highest_education": "本科",
                 "education_history": [{"school": "北京大学", "degree": "本科", "major": "AI"}]},
            ]
            mock_db.get_candidate.return_value = {
                "id": 1, "name": "张三", "highest_education": "硕士",
                "skills": [], "work_experiences": [], "projects": [], "awards_certificates": [],
                "education_history": [{"school": "清华大学", "degree": "硕士", "major": "CS"}],
            }
            mock_fusion.extract_text_features.return_value = np.random.rand(1, 1024)
            mock_vdb.search_similar.return_value = [
                {"candidate_id": 1, "distance": 0.2, "metadata": {}}
            ]
            skill = RAGRetrievalSkill()
            result = await skill.execute({"query": "清华硕士Python", "top_k": 5})
            assert "candidates" in result
            assert result["retrieval_method"] == "bm25+dense_fusion"

    def test_tokenize(self):
        from backend.skills.rag_retrieval_skill import RAGRetrievalSkill
        skill = RAGRetrievalSkill()
        tokens = skill._tokenize("Python高级工程师 Java开发")
        assert "python" in tokens
        assert "java" in tokens

    def test_compute_bm25(self):
        from backend.skills.rag_retrieval_skill import RAGRetrievalSkill
        skill = RAGRetrievalSkill()
        score = skill._compute_bm25(["python", "java"], ["python", "go", "java"], 100)
        assert score > 0
