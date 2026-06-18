"""测试API层"""
import pytest
from unittest.mock import patch, AsyncMock


class TestAPIRoutes:
    @pytest.mark.asyncio
    async def test_handle_matching(self):
        from backend.api.routes import handle_matching_request
        with patch("backend.api.routes.harness_controller") as mock_harness, \
             patch("backend.api.routes.hr_db") as mock_db:
            mock_harness.execute = AsyncMock(return_value={
                "matched_candidates": [{"candidate_id": 1, "match_score": 0.9}]
            })
            mock_db.insert_matching_history.return_value = 1
            result = await handle_matching_request("Python工程师")
            assert "matched_candidates" in result
            assert "history_id" in result

    @pytest.mark.asyncio
    async def test_handle_stats(self):
        from backend.api.routes import handle_stats
        with patch("backend.api.routes.hr_db") as mock_db:
            mock_db.get_performance_stats.return_value = {"total_queries": 5}
            mock_db.get_all_candidates_count.return_value = 100
            result = await handle_stats()
            assert "database" in result
            assert "candidates_count" in result


class TestHTTPIntegration:
    def test_process_matching_api(self):
        from backend.api.http_integration import process_matching_api
        with patch("backend.api.http_integration.handle_matching_request") as mock_handler:
            import asyncio
            mock_handler.return_value = {"matched_candidates": []}
            # 模拟异步调用
            with patch("backend.api.http_integration.asyncio.new_event_loop") as mock_loop:
                loop = mock_loop.return_value
                loop.run_until_complete.return_value = {"matched_candidates": [], "history_id": 1}
                result = process_matching_api({"query": "test"})
                assert "matched_candidates" in result

    def test_process_generate_api(self):
        from backend.api.http_integration import process_generate_api
        with patch("backend.api.http_integration.asyncio.new_event_loop") as mock_loop:
            loop = mock_loop.return_value
            loop.run_until_complete.return_value = {"generated": 50}
            result = process_generate_api({"count": 50})
            assert result == {"generated": 50}

    def test_process_preprocess_api(self):
        from backend.api.http_integration import process_preprocess_api
        with patch("backend.api.http_integration.asyncio.new_event_loop") as mock_loop:
            loop = mock_loop.return_value
            loop.run_until_complete.return_value = {"preprocess": {"processed": 10}, "vector_index": {"indexed": 10}}
            result = process_preprocess_api()
            assert "preprocess" in result

    def test_process_explain_api(self):
        from backend.api.http_integration import process_explain_api
        with patch("backend.api.http_integration.asyncio.new_event_loop") as mock_loop:
            loop = mock_loop.return_value
            loop.run_until_complete.return_value = {"global_explanation": {}}
            result = process_explain_api({"candidate_id": 1, "features": [0.5]*12, "match_score": 0.8})
            assert "global_explanation" in result

    def test_process_feedback_api(self):
        from backend.api.http_integration import process_feedback_api
        with patch("backend.api.http_integration.asyncio.new_event_loop") as mock_loop:
            loop = mock_loop.return_value
            loop.run_until_complete.return_value = {"status": "feedback_recorded"}
            result = process_feedback_api({"history_id": 1, "feedback": 1})
            assert result["status"] == "feedback_recorded"

    def test_process_stats_api(self):
        from backend.api.http_integration import process_stats_api
        with patch("backend.api.http_integration.asyncio.new_event_loop") as mock_loop:
            loop = mock_loop.return_value
            loop.run_until_complete.return_value = {"database": {}, "candidates_count": 100}
            result = process_stats_api()
            assert "database" in result
