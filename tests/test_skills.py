"""测试各Skill"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock


class TestBaseSkill:
    def test_skill_creation(self):
        from backend.skills.base_skill import BaseSkill
        class TestSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="test", description="test skill")
            async def execute(self, params):
                return {"result": "ok"}
        skill = TestSkill()
        assert skill.name == "test"
        assert skill.description == "test skill"

    @pytest.mark.asyncio
    async def test_run_success(self):
        """测试 BaseSkill.run() 正常执行路径"""
        from backend.skills.base_skill import BaseSkill
        class GoodSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="good", description="good skill")
            async def execute(self, params):
                return {"status": "done"}
        skill = GoodSkill()
        result = await skill.run({"key": "value"})
        assert result == {"status": "done"}
        assert skill.execution_count == 1
        assert skill.total_time > 0

    @pytest.mark.asyncio
    async def test_run_validates_input(self):
        """测试 BaseSkill.run() 对非dict输入的验证"""
        from backend.skills.base_skill import BaseSkill
        class SimpleSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="simple", description="simple")
            async def execute(self, params):
                return {}
        skill = SimpleSkill()
        with pytest.raises(ValueError, match="params must be dict"):
            await skill.run("not a dict")

    @pytest.mark.asyncio
    async def test_run_propagates_exception(self):
        """测试 BaseSkill.run() 在execute抛异常时重新抛出"""
        from backend.skills.base_skill import BaseSkill
        class FailSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="fail", description="fail")
            async def execute(self, params):
                raise RuntimeError("Something broke")
        skill = FailSkill()
        with pytest.raises(RuntimeError, match="Something broke"):
            await skill.run({})
        # execution_count should NOT increment on failure
        assert skill.execution_count == 0

    def test_get_stats(self):
        """测试 BaseSkill.get_stats()"""
        from backend.skills.base_skill import BaseSkill
        class StatsSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="stats", description="stats")
            async def execute(self, params):
                return {}
        skill = StatsSkill()
        stats = skill.get_stats()
        assert stats["name"] == "stats"
        assert stats["executions"] == 0
        assert stats["avg_time"] == 0.0

    @pytest.mark.asyncio
    async def test_run_increments_stats_correctly(self):
        """测试多次执行后统计正确"""
        from backend.skills.base_skill import BaseSkill
        class CountSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="count", description="count")
            async def execute(self, params):
                return {"ok": True}
        skill = CountSkill()
        await skill.run({})
        await skill.run({})
        await skill.run({})
        assert skill.execution_count == 3
        stats = skill.get_stats()
        assert stats["executions"] == 3
        assert stats["avg_time"] >= 0


class TestJDParserSkill:
    @pytest.mark.asyncio
    async def test_parse(self):
        from backend.skills.jd_parser_skill import JDParserSkill
        with patch("backend.skills.jd_parser_skill.chat_json") as mock_chat:
            mock_chat.return_value = {
                "hard_rules": [{"field": "education", "operator": "=", "value": "硕士"}],
                "soft_requirements": [],
                "summary": "Python工程师",
                "required_skills": ["Python"],
                "education_req": "硕士",
                "min_experience": 3,
            }
            skill = JDParserSkill()
            result = await skill.execute({"query": "招聘Python高级工程师"})
            assert "hard_rules" in result
            assert "summary" in result


class TestResumeGeneratorSkill:
    @pytest.mark.asyncio
    async def test_generate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.config.DB_PATH", str(tmp_path / "test.db"))
        from backend.database.models import HRDatabase
        with patch("backend.database.models.hr_db") as mock_db:
            mock_db.insert_candidate.return_value = 1
            mock_db.insert_skill.return_value = 1
            mock_db.insert_work_experience.return_value = 1
            mock_db.insert_project.return_value = 1
            mock_db.insert_award_certificate.return_value = 1
            mock_db.insert_education_history.return_value = 1
            mock_db.get_all_candidates_count.return_value = 5
            from backend.skills.resume_generator_skill import ResumeGeneratorSkill
            skill = ResumeGeneratorSkill()
            result = await skill.execute({"count": 5})
            assert result["generated"] == 5


class TestDataPreprocessingSkill:
    @pytest.mark.asyncio
    async def test_preprocess(self):
        from backend.skills.data_preprocessing_skill import DataPreprocessingSkill
        with patch("backend.skills.data_preprocessing_skill.hr_db") as mock_db:
            mock_db.search_candidates.return_value = [
                {"id": 1, "name": "张三", "age": 28, "work_years": 5,
                 "highest_education": "硕士"}
            ]
            skill = DataPreprocessingSkill()
            result = await skill.execute({"action": "preprocess_all"})
            assert result["processed"] == 1

    @pytest.mark.asyncio
    async def test_preprocess_cleans_invalid_age(self):
        from backend.skills.data_preprocessing_skill import DataPreprocessingSkill
        with patch("backend.skills.data_preprocessing_skill.hr_db") as mock_db:
            cand = {"id": 1, "name": "测试", "age": 100, "work_years": -1}
            mock_db.search_candidates.return_value = [cand]
            skill = DataPreprocessingSkill()
            await skill.execute({"action": "preprocess_all"})
            # After cleaning, age > 65 should be None, work_years < 0 should be 0
            assert cand["age"] is None
            assert cand["work_years"] == 0

    @pytest.mark.asyncio
    async def test_build_vectors(self):
        import numpy as np
        from backend.skills.data_preprocessing_skill import DataPreprocessingSkill
        with patch("backend.skills.data_preprocessing_skill.hr_db") as mock_db, \
             patch("backend.skills.data_preprocessing_skill.multimodal_fusion") as mock_fusion, \
             patch("backend.skills.data_preprocessing_skill.vector_db") as mock_vdb:
            mock_db.search_candidates.return_value = [{"id": 1}, {"id": 2}]
            mock_db.get_candidate.side_effect = [
                {"id": 1, "name": "张三", "highest_education": "硕士",
                 "work_years": 5, "skills": [{"skill_name": "Python"}],
                 "work_experiences": [{"company_name": "美团", "position": "工程师"}],
                 "projects": [{"project_name": "AI项目", "technologies": "Python,TF"}],
                 "education_history": [{"school": "清华大学", "degree": "硕士", "major": "计算机"}],
                 "awards_certificates": []},
                None,  # Second candidate not found
            ]
            mock_fusion.extract_text_features.return_value = np.random.rand(1, 1024)
            mock_vdb.get_collection_count.return_value = 1
            mock_vdb.add_candidate.return_value = None

            skill = DataPreprocessingSkill()
            result = await skill.execute({"action": "build_vectors"})
            assert result["indexed"] == 1  # Only 1 indexed (second was None)
            assert result["total_vectors"] == 1


class TestMatchingEvaluationSkill:
    @pytest.mark.asyncio
    async def test_matching(self):
        from backend.skills.matching_evaluation_skill import MatchingEvaluationSkill
        with patch("backend.skills.matching_evaluation_skill.hr_db") as mock_db, \
             patch("backend.skills.matching_evaluation_skill.multimodal_fusion") as mock_fusion, \
             patch("backend.skills.matching_evaluation_skill.catboost_matcher") as mock_cb:
            mock_db.get_candidate.return_value = {
                "id": 1, "name": "张三", "highest_education": "硕士", "work_years": 5,
                "skills": [{"skill_name": "Python"}],
                "work_experiences": [{"company_name": "美团", "position": "高级工程师"}],
                "projects": [], "awards_certificates": [],
            }
            mock_fusion.compute_matching_score.return_value = {
                "score": 0.85, "text_similarity": 0.8, "multimodal_similarity": 0.0
            }
            mock_cb.extract_structured_features.return_value = __import__("numpy").random.rand(12)
            mock_cb.predict.return_value = 0.82

            skill = MatchingEvaluationSkill()
            result = await skill.execute({
                "query": "Python工程师",
                "candidates": {"candidates": [{"candidate_id": 1, "data": {"id": 1}}]},
                "jd_info": {"hard_rules": []},
            })
            assert "matched_candidates" in result


class TestSHAPExplainerSkill:
    @pytest.mark.asyncio
    async def test_explain_all(self):
        from backend.skills.shap_explainer_skill import SHAPExplainerSkill
        skill = SHAPExplainerSkill()
        result = await skill.execute({
            "candidate_id": 1,
            "features": [0.8, 0.9, 0.7, 0.6, 0.5, 0.8, 0.7, 0.4, 0.3, 0.5, 0.6, 0.7],
            "match_score": 0.85,
            "level": "all",
        })
        assert "global_explanation" in result
        assert "individual_explanation" in result
        assert "interaction_explanation" in result
        assert "nlp_explanation" in result

    @pytest.mark.asyncio
    async def test_nlp_explanation(self):
        from backend.skills.shap_explainer_skill import SHAPExplainerSkill
        skill = SHAPExplainerSkill()
        result = await skill.execute({
            "candidate_id": 2,
            "features": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.6, 0.7],
            "match_score": 0.78,
            "level": "nlp",
            "detail": True,
        })
        assert "nlp_explanation" in result
        assert "explanation" in result["nlp_explanation"]


class TestDatabaseOperationSkill:
    @pytest.mark.asyncio
    async def test_search_action(self):
        from backend.skills.database_operation_skill import DatabaseOperationSkill
        with patch("backend.skills.database_operation_skill.hr_db") as mock_db:
            mock_db.search_candidates.return_value = [{"id": 1, "name": "张三"}]
            skill = DatabaseOperationSkill()
            result = await skill.execute({"action": "search", "limit": 10})
            assert "candidates" in result
            assert len(result["candidates"]) == 1

    @pytest.mark.asyncio
    async def test_get_action(self):
        from backend.skills.database_operation_skill import DatabaseOperationSkill
        with patch("backend.skills.database_operation_skill.hr_db") as mock_db:
            mock_db.get_candidate.return_value = {"id": 1, "name": "张三"}
            skill = DatabaseOperationSkill()
            result = await skill.execute({"action": "get", "candidate_id": 1})
            assert result["candidate"] == {"id": 1, "name": "张三"}

    @pytest.mark.asyncio
    async def test_get_action_no_id(self):
        from backend.skills.database_operation_skill import DatabaseOperationSkill
        with patch("backend.skills.database_operation_skill.hr_db") as mock_db:
            skill = DatabaseOperationSkill()
            result = await skill.execute({"action": "get"})
            assert result["candidate"] is None

    @pytest.mark.asyncio
    async def test_count_action(self):
        from backend.skills.database_operation_skill import DatabaseOperationSkill
        with patch("backend.skills.database_operation_skill.hr_db") as mock_db:
            mock_db.get_all_candidates_count.return_value = 42
            skill = DatabaseOperationSkill()
            result = await skill.execute({"action": "count"})
            assert result["count"] == 42

    @pytest.mark.asyncio
    async def test_stats_action(self):
        from backend.skills.database_operation_skill import DatabaseOperationSkill
        with patch("backend.skills.database_operation_skill.hr_db") as mock_db:
            mock_db.get_performance_stats.return_value = {"total_queries": 10}
            skill = DatabaseOperationSkill()
            result = await skill.execute({"action": "stats"})
            assert result["stats"] == {"total_queries": 10}

    @pytest.mark.asyncio
    async def test_vector_search_action(self):
        from backend.skills.database_operation_skill import DatabaseOperationSkill
        with patch("backend.skills.database_operation_skill.vector_db") as mock_vdb:
            mock_vdb.search_similar.return_value = [{"candidate_id": 1, "distance": 0.1}]
            skill = DatabaseOperationSkill()
            result = await skill.execute({"action": "vector_search", "embedding": [0.1]*10, "top_k": 5})
            assert "results" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        from backend.skills.database_operation_skill import DatabaseOperationSkill
        with patch("backend.skills.database_operation_skill.hr_db"):
            skill = DatabaseOperationSkill()
            result = await skill.execute({"action": "invalid_action"})
            assert "error" in result

    @pytest.mark.asyncio
    async def test_default_action_is_search(self):
        from backend.skills.database_operation_skill import DatabaseOperationSkill
        with patch("backend.skills.database_operation_skill.hr_db") as mock_db:
            mock_db.search_candidates.return_value = []
            skill = DatabaseOperationSkill()
            result = await skill.execute({})
            assert "candidates" in result


class TestFeedbackLearningSkill:
    @pytest.mark.asyncio
    async def test_submit_feedback(self):
        from backend.skills.feedback_learning_skill import FeedbackLearningSkill
        with patch("backend.skills.feedback_learning_skill.hr_db") as mock_db:
            mock_db.update_feedback.return_value = None
            mock_db.get_recent_feedback.return_value = []
            skill = FeedbackLearningSkill()
            result = await skill.execute({
                "action": "submit_feedback",
                "history_id": 1,
                "feedback": 1,
            })
            assert result["status"] == "feedback_recorded"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        from backend.skills.feedback_learning_skill import FeedbackLearningSkill
        with patch("backend.skills.feedback_learning_skill.hr_db") as mock_db, \
             patch("backend.skills.feedback_learning_skill.dynamic_scheduler") as mock_sched, \
             patch("backend.skills.feedback_learning_skill.catboost_matcher") as mock_cb:
            mock_db.get_performance_stats.return_value = {"total_queries": 10}
            mock_sched.get_stats.return_value = {"quality_threshold": 0.75}
            mock_cb.is_trained = False
            skill = FeedbackLearningSkill()
            result = await skill.execute({"action": "get_stats"})
            assert "db_stats" in result
