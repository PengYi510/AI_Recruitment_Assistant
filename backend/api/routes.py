"""API路由 - 整合Harness驱动的智能匹配"""
import time, logging, json
from typing import Dict, Any
from backend.harness.harness import harness as harness_controller
from backend.database.models import hr_db
from backend.skills.resume_generator_skill import ResumeGeneratorSkill
from backend.skills.data_preprocessing_skill import DataPreprocessingSkill
from backend.skills.shap_explainer_skill import SHAPExplainerSkill
from backend.skills.feedback_learning_skill import FeedbackLearningSkill

logger = logging.getLogger(__name__)


async def handle_matching_request(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """处理匹配请求 - Harness驱动流程"""
    start_time = time.time()
    try:
        result = await harness_controller.execute(query, context or {})
        latency_ms = (time.time() - start_time) * 1000

        # 保存匹配历史
        candidates = result.get("matched_candidates", [])
        cand_ids = [c.get("candidate_id", 0) for c in candidates]
        scores = [c.get("match_score", 0) for c in candidates]
        history_id = hr_db.insert_matching_history(query, cand_ids, scores, latency_ms)

        result["history_id"] = history_id
        result["latency_ms"] = round(latency_ms, 2)
        return result
    except Exception as e:
        logger.error(f"Matching failed: {e}")
        return {"error": str(e), "matched_candidates": []}


async def handle_generate_data(count: int = 100) -> Dict[str, Any]:
    """生成合成简历数据"""
    skill = ResumeGeneratorSkill()
    return await skill.execute({"count": count})


async def handle_preprocess() -> Dict[str, Any]:
    """预处理数据并构建向量索引"""
    skill = DataPreprocessingSkill()
    r1 = await skill.execute({"action": "preprocess_all"})
    r2 = await skill.execute({"action": "build_vectors"})
    return {"preprocess": r1, "vector_index": r2}


async def handle_explain(candidate_id: int, features=None, match_score=0.0, level="all") -> Dict[str, Any]:
    """获取SHAP可解释性分析"""
    skill = SHAPExplainerSkill()
    return await skill.execute({
        "candidate_id": candidate_id,
        "features": features,
        "match_score": match_score,
        "level": level,
    })


async def handle_feedback(history_id: int, feedback: int) -> Dict[str, Any]:
    """提交反馈"""
    skill = FeedbackLearningSkill()
    return await skill.execute({
        "action": "submit_feedback",
        "history_id": history_id,
        "feedback": feedback,
    })


async def handle_stats() -> Dict[str, Any]:
    """获取系统统计"""
    return {
        "database": hr_db.get_performance_stats(),
        "candidates_count": hr_db.get_all_candidates_count(),
    }
