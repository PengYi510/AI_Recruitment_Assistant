"""测试Harness框架"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock


class TestDynamicScheduler:
    def test_evaluate_complexity(self):
        from backend.harness.dynamic_scheduler import DynamicScheduler
        with patch("backend.harness.dynamic_scheduler.chat_json") as mock_chat:
            mock_chat.return_value = {"complexity": 0.3, "reason": "simple"}
            scheduler = DynamicScheduler()
            simple = scheduler.evaluate_complexity("找Python工程师")
            assert 0 <= simple <= 1

    def test_heuristic_complexity(self):
        from backend.harness.dynamic_scheduler import DynamicScheduler
        scheduler = DynamicScheduler()
        # 简单查询
        simple = scheduler._heuristic_complexity("查找Python工程师")
        assert 0 <= simple <= 1
        # 复杂查询 (longer + complex keywords)
        complex_q = ("需要一个精通Python、Java、Go的高级架构师," * 5 +
                    "对比分析排名推荐匹配解释详细")
        complex_score = scheduler._heuristic_complexity(complex_q)
        assert complex_score > simple

    def test_split_subtasks(self):
        from backend.harness.dynamic_scheduler import DynamicScheduler
        with patch("backend.harness.dynamic_scheduler.chat_json") as mock_chat:
            mock_chat.return_value = {"subtasks": [
                {"description": "解析需求"},
                {"description": "检索候选人"},
                {"description": "匹配评分"}
            ]}
            scheduler = DynamicScheduler()
            subtasks = scheduler.split_subtasks("找Python高级工程师", 0.8)
            assert len(subtasks) >= 1
            assert all(hasattr(st, "description") for st in subtasks)

    def test_determine_iterations(self):
        from backend.harness.dynamic_scheduler import DynamicScheduler
        scheduler = DynamicScheduler()
        assert scheduler.determine_iterations(0.1) == 1  # simple
        assert scheduler.determine_iterations(0.5) == 2  # medium
        assert scheduler.determine_iterations(0.8) == 3  # complex

    def test_record_feedback(self):
        from backend.harness.dynamic_scheduler import DynamicScheduler
        scheduler = DynamicScheduler()
        for i in range(15):
            scheduler.record_feedback(0.5, True, 1, 0.5)
        assert len(scheduler.feedback_history) == 15

    def test_get_stats_empty(self):
        from backend.harness.dynamic_scheduler import DynamicScheduler
        scheduler = DynamicScheduler()
        stats = scheduler.get_stats()
        assert "total_tasks" in stats
        assert stats["total_tasks"] == 0

    def test_get_stats_with_data(self):
        from backend.harness.dynamic_scheduler import DynamicScheduler
        scheduler = DynamicScheduler()
        scheduler.record_feedback(0.5, True, 1, 1.0)
        scheduler.record_feedback(0.8, False, 3, 2.0)
        stats = scheduler.get_stats()
        assert stats["total_tasks"] == 2
        assert "success_rate" in stats
        assert "avg_response_time" in stats


class TestHarnessState:
    def test_task_state_creation(self):
        from backend.harness.state import TaskState, TaskStatus
        state = TaskState(task_id="test-1", user_query="找工程师")
        assert state.status == TaskStatus.PENDING
        assert state.current_iteration == 0

    def test_task_status_transitions(self):
        from backend.harness.state import TaskState, TaskStatus
        state = TaskState(task_id="test-2", user_query="测试")
        state.status = TaskStatus.PLANNING
        assert state.status == TaskStatus.PLANNING
        state.mark_completed()
        assert state.status == TaskStatus.COMPLETED

    def test_task_mark_failed(self):
        from backend.harness.state import TaskState, TaskStatus
        state = TaskState(task_id="test-3", user_query="测试")
        state.mark_failed("some error")
        assert state.status == TaskStatus.FAILED
        assert "some error" in state.error_log

    def test_subtask_creation(self):
        from backend.harness.state import SubTask, TaskStatus
        st = SubTask(task_id="sub-1", description="解析JD")
        assert st.status == TaskStatus.PENDING
        assert st.description == "解析JD"


class TestHarnessController:
    @pytest.mark.asyncio
    async def test_execute_simple_query(self):
        from backend.harness.harness import HarnessController
        with patch("backend.harness.dynamic_scheduler.chat_json") as mock_sched_chat, \
             patch.object(HarnessController, '__init__', lambda self: None):
            mock_sched_chat.return_value = {"complexity": 0.3, "reason": "simple"}

            controller = HarnessController.__new__(HarnessController)
            controller.planner = MagicMock()
            controller.generator = MagicMock()
            controller.evaluator = MagicMock()

            controller.planner.plan = AsyncMock(return_value={"task_type": "match", "steps": ["search"]})
            controller.generator.generate = AsyncMock(return_value={"matched_candidates": [{"id": 1}]})
            controller.evaluator.evaluate = AsyncMock(return_value={"passed": True, "score": 0.9})

            with patch("backend.harness.harness.dynamic_scheduler") as mock_ds, \
                 patch("backend.harness.harness.hr_db") as mock_db:
                from backend.harness.state import TaskState, SubTask
                mock_state = TaskState(task_id="t1", user_query="test", start_time=1.0)
                mock_state.sub_tasks = [SubTask(task_id="s1", description="test")]
                mock_ds.create_execution_plan.return_value = mock_state
                mock_ds.record_feedback.return_value = None
                mock_db.save_system_feedback.return_value = None

                result = await controller.execute("找Python工程师", {})
                assert isinstance(result, dict)
"""
"""
