"""Generator Agent - 任务执行与结果生成（支持动态特征评分）"""
import logging
from typing import Dict, Any
from backend.harness.state import TaskState, SubTask
from backend.skills.skill_registry import skill_registry
from backend.models.catboost_matcher import catboost_matcher

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """生成Agent: 调用Skills执行任务,生成初步结果"""

    async def generate(self, subtask: SubTask, state: TaskState, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行子任务,调用相应Skills"""
        context = context or {}
        plan = state.plan
        task_type = plan.get("task_type", "candidate_search")
        skills_needed = plan.get("skills_needed", [])

        logger.info(f"[Generator] Executing subtask: {subtask.description[:50]}...")

        result = {"subtask_id": subtask.task_id, "status": "completed"}

        try:
            # 根据任务类型调用对应的Skills
            if task_type == "jd_parse":
                skill = skill_registry.get_skill("jd_parser")
                if skill:
                    result["data"] = await skill.execute({"query": subtask.description, "context": context})

            elif task_type == "candidate_search":
                # 候选人搜索类任务：始终使用完整原始 query 来调用 RAG 和 Matching
                # 这确保所有硬约束（985/应届生/全日制/学历等）不会因子任务分解而丢失
                full_query = state.user_query
                
                # RAG检索 -> 匹配评估 -> SHAP可解释性
                rag_skill = skill_registry.get_skill("rag_retrieval")
                if rag_skill:
                    candidates = await rag_skill.execute({"query": full_query, "context": context})
                    result["candidates"] = candidates

                match_skill = skill_registry.get_skill("matching_evaluation")
                if match_skill and result.get("candidates"):
                    scored = await match_skill.execute({
                        "query": full_query,
                        "candidates": result["candidates"],
                        "context": context
                    })
                    result["matched"] = scored

                    # SHAP可解释性分析：为每个匹配候选人计算SHAP值（含动态特征）
                    shap_skill = skill_registry.get_skill("shap_explainer")
                    if shap_skill and scored.get("matched_candidates"):
                        # 从 RAG 结果中获取 LLM 提取的动态约束
                        extra_constraints = candidates.get("constraints_detected", {}).get("_extra_constraints", {})
                        
                        shap_results = []
                        for mc in scored["matched_candidates"][:10]:
                            # 计算动态特征分数（如 GPA匹配度、目标岗位匹配度）
                            dynamic_scores = {}
                            if extra_constraints:
                                dynamic_scores = catboost_matcher.compute_dynamic_feature_scores(
                                    mc.get("candidate_data", mc), extra_constraints
                                )
                            
                            shap_out = await shap_skill.execute({
                                "candidate_id": mc.get("candidate_id"),
                                "features": mc.get("structured_features"),
                                "match_score": mc.get("match_score", 0),
                                "level": "individual",
                                "query": full_query,
                                "extra_constraints": extra_constraints,
                                "dynamic_scores": dynamic_scores,
                            })
                            shap_results.append({
                                "candidate_id": mc.get("candidate_id"),
                                "shap_values": shap_out.get("individual_explanation", {}).get("shap_values", {}),
                                "base_value": shap_out.get("individual_explanation", {}).get("base_value", 0.5),
                                "dynamic_feature_count": shap_out.get("individual_explanation", {}).get("dynamic_feature_count", 0),
                            })
                        result["shap_explanations"] = shap_results

            elif task_type == "matching":
                match_skill = skill_registry.get_skill("matching_evaluation")
                if match_skill:
                    result["data"] = await match_skill.execute({"query": subtask.description, "context": context})

            elif task_type == "explanation":
                shap_skill = skill_registry.get_skill("shap_explainer")
                if shap_skill:
                    result["explanation"] = await shap_skill.execute({"query": subtask.description, "context": context})

            elif task_type == "data_generate":
                gen_skill = skill_registry.get_skill("resume_generator")
                if gen_skill:
                    # 支持从 context/plan 指定生成份数，默认生成 1 份
                    gen_count = context.get("count") or plan.get("count") or 1
                    result["data"] = await gen_skill.execute({"count": int(gen_count)})

            elif task_type == "feedback":
                fb_skill = skill_registry.get_skill("feedback_learning")
                if fb_skill:
                    result["data"] = await fb_skill.execute({"query": subtask.description, "context": context})
            else:
                # 默认使用RAG检索
                rag_skill = skill_registry.get_skill("rag_retrieval")
                if rag_skill:
                    result["data"] = await rag_skill.execute({"query": subtask.description, "context": context})

        except Exception as e:
            logger.error(f"[Generator] Error: {e}")
            result["status"] = "partial"
            result["error"] = str(e)

        return result
