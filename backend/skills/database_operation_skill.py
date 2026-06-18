"""数据库操作Skill - SQLite和ChromaDB的增删改查"""
import logging
from typing import Dict, Any
from backend.skills.base_skill import BaseSkill
from backend.database.models import hr_db
from backend.vector_db.client import vector_db

logger = logging.getLogger(__name__)


class DatabaseOperationSkill(BaseSkill):
    """数据库操作Skill"""

    def __init__(self):
        super().__init__(name="database_operation", description="SQLite和ChromaDB增删改查")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "search")
        if action == "search":
            return {"candidates": hr_db.search_candidates(params.get("filters"), params.get("limit", 50))}
        elif action == "get":
            cid = params.get("candidate_id")
            return {"candidate": hr_db.get_candidate(cid) if cid else None}
        elif action == "count":
            return {"count": hr_db.get_all_candidates_count()}
        elif action == "stats":
            return {"stats": hr_db.get_performance_stats()}
        elif action == "vector_search":
            embedding = params.get("embedding", [])
            top_k = params.get("top_k", 20)
            return {"results": vector_db.search_similar(embedding, top_k)}
        return {"error": f"Unknown action: {action}"}
