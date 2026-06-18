"""反馈学习Skill - 基于用户反馈动态调整模型权重"""
import logging
from typing import Dict, Any
from backend.skills.base_skill import BaseSkill
from backend.database.models import hr_db
from backend.models.catboost_matcher import catboost_matcher
from backend.harness.dynamic_scheduler import dynamic_scheduler

logger = logging.getLogger(__name__)


class FeedbackLearningSkill(BaseSkill):
    """反馈学习Skill: 基于用户反馈调整CatBoost特征权重"""

    def __init__(self):
        super().__init__(name="feedback_learning", description="基于反馈动态调整模型权重")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "submit_feedback")
        if action == "submit_feedback":
            return await self._process_feedback(params)
        elif action == "get_stats":
            return self._get_learning_stats()
        return {"error": "Unknown action"}

    async def _process_feedback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户反馈"""
        history_id = params.get("history_id")
        feedback = params.get("feedback")  # 1=满意, 0=不满意
        query = params.get("query", "")

        if history_id and feedback is not None:
            hr_db.update_feedback(history_id, feedback)

        # 获取最近反馈统计
        recent = hr_db.get_recent_feedback(50)
        if len(recent) >= 20:
            # 分析反馈模式,调整权重
            positive = sum(1 for r in recent if r.get("success"))
            negative = len(recent) - positive
            if negative > positive * 0.3:
                # 负面反馈过多,触发权重调整
                catboost_matcher.update_weights({"adjustment": "increase_diversity"})

        return {"status": "feedback_recorded", "history_id": history_id}

    def _get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        stats = hr_db.get_performance_stats()
        scheduler_stats = dynamic_scheduler.get_stats()
        return {
            "db_stats": stats,
            "scheduler_stats": scheduler_stats,
            "model_trained": catboost_matcher.is_trained,
        }
