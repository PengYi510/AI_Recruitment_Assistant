"""HTTP集成层 - 与现有http_server.py对接"""
import json, asyncio, logging
from typing import Dict, Any
from backend.api.routes import (
    handle_matching_request, handle_generate_data,
    handle_preprocess, handle_explain,
    handle_feedback, handle_stats,
)

logger = logging.getLogger(__name__)


def process_matching_api(request_body: Dict[str, Any]) -> Dict[str, Any]:
    """同步包装 - 供http_server.py调用"""
    query = request_body.get("query", request_body.get("message", ""))
    context = request_body.get("context", {})
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(handle_matching_request(query, context))
        return result
    finally:
        loop.close()


def process_generate_api(request_body: Dict[str, Any]) -> Dict[str, Any]:
    count = request_body.get("count", 100)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(handle_generate_data(count))
    finally:
        loop.close()


def process_preprocess_api() -> Dict[str, Any]:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(handle_preprocess())
    finally:
        loop.close()


def process_explain_api(request_body: Dict[str, Any]) -> Dict[str, Any]:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(handle_explain(
            candidate_id=request_body.get("candidate_id", 0),
            features=request_body.get("features"),
            match_score=request_body.get("match_score", 0.0),
            level=request_body.get("level", "all")
        ))
    finally:
        loop.close()


def process_feedback_api(request_body: Dict[str, Any]) -> Dict[str, Any]:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(handle_feedback(
            history_id=request_body.get("history_id", 0),
            feedback=request_body.get("feedback", 0)
        ))
    finally:
        loop.close()


def process_stats_api() -> Dict[str, Any]:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(handle_stats())
    finally:
        loop.close()
