"""测试Agent"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from backend.harness.state import TaskState, SubTask, TaskStatus


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan(self):
        from backend.agents.planner_agent import PlannerAgent
        with patch("backend.agents.planner_agent.chat_json") as mock_chat:
            mock_chat.return_value = {
                "task_type": "candidate_search",
                "skills_needed": ["rag_retrieval", "matching_evaluation"],
                "acceptance_criteria": "返回相关候选人",
                "priority": "medium",
                "requires_multimodal": False
            }
            agent = PlannerAgent()
            state = TaskState(task_id="t1", user_query="找Python工程师")
            result = await agent.plan("找Python工程师", state, {})
            assert "task_type" in result
            assert "skills_needed" in result

    @pytest.mark.asyncio
    async def test_plan_fallback(self):
        from backend.agents.planner_agent import PlannerAgent
        with patch("backend.agents.planner_agent.chat_json") as mock_chat:
            mock_chat.side_effect = Exception("LLM unavailable")
            agent = PlannerAgent()
            state = TaskState(task_id="t2", user_query="测试")
            result = await agent.plan("测试", state, {})
            # Should fall back to default plan
            assert result["task_type"] == "candidate_search"


class TestGeneratorAgent:
    @pytest.mark.asyncio
    async def test_generate(self):
        from backend.agents.generator_agent import GeneratorAgent
        with patch("backend.agents.generator_agent.skill_registry") as mock_reg:
            mock_skill = MagicMock()
            mock_skill.execute = AsyncMock(return_value={"candidates": [{"id": 1}]})
            mock_reg.get_skill.return_value = mock_skill

            agent = GeneratorAgent()
            subtask = SubTask(task_id="s1", description="找Python工程师")
            state = TaskState(task_id="t1", user_query="找Python工程师")
            state.plan = {"task_type": "candidate_search", "skills_needed": ["rag_retrieval"]}

            result = await agent.generate(subtask, state, {})
            assert isinstance(result, dict)
            assert "subtask_id" in result

    @pytest.mark.asyncio
    async def test_generate_jd_parse(self):
        from backend.agents.generator_agent import GeneratorAgent
        with patch("backend.agents.generator_agent.skill_registry") as mock_reg:
            mock_skill = MagicMock()
            mock_skill.execute = AsyncMock(return_value={"hard_rules": [], "summary": "test"})
            mock_reg.get_skill.return_value = mock_skill
            agent = GeneratorAgent()
            subtask = SubTask(task_id="s1", description="解析JD")
            state = TaskState(task_id="t1", user_query="解析JD")
            state.plan = {"task_type": "jd_parse", "skills_needed": ["jd_parser"]}
            result = await agent.generate(subtask, state)
            assert result["status"] == "completed"
            assert "data" in result

    @pytest.mark.asyncio
    async def test_generate_matching(self):
        from backend.agents.generator_agent import GeneratorAgent
        with patch("backend.agents.generator_agent.skill_registry") as mock_reg:
            mock_skill = MagicMock()
            mock_skill.execute = AsyncMock(return_value={"matched_candidates": []})
            mock_reg.get_skill.return_value = mock_skill
            agent = GeneratorAgent()
            subtask = SubTask(task_id="s1", description="匹配")
            state = TaskState(task_id="t1", user_query="匹配")
            state.plan = {"task_type": "matching", "skills_needed": ["matching_evaluation"]}
            result = await agent.generate(subtask, state)
            assert result["status"] == "completed"
            assert "data" in result

    @pytest.mark.asyncio
    async def test_generate_explanation(self):
        from backend.agents.generator_agent import GeneratorAgent
        with patch("backend.agents.generator_agent.skill_registry") as mock_reg:
            mock_skill = MagicMock()
            mock_skill.execute = AsyncMock(return_value={"explanation": "test"})
            mock_reg.get_skill.return_value = mock_skill
            agent = GeneratorAgent()
            subtask = SubTask(task_id="s1", description="解释")
            state = TaskState(task_id="t1", user_query="解释")
            state.plan = {"task_type": "explanation", "skills_needed": ["shap_explainer"]}
            result = await agent.generate(subtask, state)
            assert "explanation" in result

    @pytest.mark.asyncio
    async def test_generate_data_generate(self):
        from backend.agents.generator_agent import GeneratorAgent
        with patch("backend.agents.generator_agent.skill_registry") as mock_reg:
            mock_skill = MagicMock()
            mock_skill.execute = AsyncMock(return_value={"generated": 10})
            mock_reg.get_skill.return_value = mock_skill
            agent = GeneratorAgent()
            subtask = SubTask(task_id="s1", description="生成数据")
            state = TaskState(task_id="t1", user_query="生成")
            state.plan = {"task_type": "data_generate", "skills_needed": ["resume_generator"]}
            result = await agent.generate(subtask, state)
            assert "data" in result

    @pytest.mark.asyncio
    async def test_generate_feedback(self):
        from backend.agents.generator_agent import GeneratorAgent
        with patch("backend.agents.generator_agent.skill_registry") as mock_reg:
            mock_skill = MagicMock()
            mock_skill.execute = AsyncMock(return_value={"status": "ok"})
            mock_reg.get_skill.return_value = mock_skill
            agent = GeneratorAgent()
            subtask = SubTask(task_id="s1", description="反馈")
            state = TaskState(task_id="t1", user_query="反馈")
            state.plan = {"task_type": "feedback", "skills_needed": ["feedback_learning"]}
            result = await agent.generate(subtask, state)
            assert "data" in result

    @pytest.mark.asyncio
    async def test_generate_unknown_type_defaults_to_rag(self):
        from backend.agents.generator_agent import GeneratorAgent
        with patch("backend.agents.generator_agent.skill_registry") as mock_reg:
            mock_skill = MagicMock()
            mock_skill.execute = AsyncMock(return_value={"results": []})
            mock_reg.get_skill.return_value = mock_skill
            agent = GeneratorAgent()
            subtask = SubTask(task_id="s1", description="未知")
            state = TaskState(task_id="t1", user_query="未知")
            state.plan = {"task_type": "unknown_type", "skills_needed": []}
            result = await agent.generate(subtask, state)
            assert "data" in result

    @pytest.mark.asyncio
    async def test_generate_handles_exception(self):
        from backend.agents.generator_agent import GeneratorAgent
        with patch("backend.agents.generator_agent.skill_registry") as mock_reg:
            mock_skill = MagicMock()
            mock_skill.execute = AsyncMock(side_effect=RuntimeError("skill failed"))
            mock_reg.get_skill.return_value = mock_skill
            agent = GeneratorAgent()
            subtask = SubTask(task_id="s1", description="失败")
            state = TaskState(task_id="t1", user_query="失败")
            state.plan = {"task_type": "jd_parse", "skills_needed": ["jd_parser"]}
            result = await agent.generate(subtask, state)
            assert result["status"] == "partial"
            assert "error" in result


class TestEvaluatorAgent:
    @pytest.mark.asyncio
    async def test_evaluate_pass(self):
        from backend.agents.evaluator_agent import EvaluatorAgent
        agent = EvaluatorAgent()
        subtask = SubTask(task_id="s1", description="找工程师")
        state = TaskState(task_id="t1", user_query="找工程师")
        state.plan = {"acceptance_criteria": "返回候选人"}
        state.complexity_score = 0.3

        result = await agent.evaluate(
            {"status": "completed", "candidates": [{"id": 1}]},
            subtask, state
        )
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_evaluate_empty_result(self):
        from backend.agents.evaluator_agent import EvaluatorAgent
        agent = EvaluatorAgent()
        subtask = SubTask(task_id="s1", description="找工程师")
        state = TaskState(task_id="t1", user_query="找工程师")
        state.plan = {"acceptance_criteria": "返回候选人"}

        result = await agent.evaluate(None, subtask, state)
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_evaluate_error_result(self):
        from backend.agents.evaluator_agent import EvaluatorAgent
        agent = EvaluatorAgent()
        subtask = SubTask(task_id="s1", description="找工程师")
        state = TaskState(task_id="t1", user_query="找工程师")
        state.plan = {"acceptance_criteria": "返回候选人"}

        result = await agent.evaluate(
            {"error": "timeout", "status": "error"},
            subtask, state
        )
        assert result["passed"] is False
