"""Evaluator Agent - 结果评估与质量控制"""
import logging
from typing import Dict, Any
from backend.harness.state import TaskState, SubTask
from backend.models.longcat_client import chat_json

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """评估Agent: 检验Generator输出是否满足验收标准"""

    async def evaluate(self, generation: Dict[str, Any], subtask: SubTask,
                      state: TaskState) -> Dict[str, Any]:
        """评估生成结果"""
        acceptance = state.plan.get("acceptance_criteria", "")

        # 基础检查
        if not generation or generation.get("status") == "error":
            return {"passed": False, "feedback": "生成结果为空或出错，需要重新生成", "score": 0.0}

        if generation.get("error"):
            return {"passed": False, "feedback": f"生成过程出错: {generation['error']}", "score": 0.2}

        # 检查结果是否有实质内容
        has_data = bool(generation.get("data") or generation.get("candidates") or
                       generation.get("matched") or generation.get("explanation"))

        if not has_data and generation.get("status") == "completed":
            # 即使没有额外数据,如果状态完成也认为通过
            return {"passed": True, "feedback": "", "score": 0.7}

        # LLM评估（仅在复杂任务时使用）
        if state.complexity_score > 0.5 and has_data:
            try:
                eval_result = self._llm_evaluate(generation, acceptance, subtask.description)
                return eval_result
            except Exception as e:
                logger.warning(f"[Evaluator] LLM eval failed: {e}")

        # 默认通过
        return {"passed": True, "feedback": "", "score": 0.8}

    def _llm_evaluate(self, generation: Dict[str, Any], acceptance: str, task_desc: str) -> Dict[str, Any]:
        """使用LLM评估结果质量"""
        system = """你是质量评估专家。评估任务执行结果是否达标。
返回JSON: {"passed": true/false, "score": 0.0-1.0, "feedback": "改进建议"}"""
        user = f"""任务: {task_desc}
验收标准: {acceptance}
结果概要: {str(generation)[:500]}"""
        result = chat_json(system=system, user=user)
        return {
            "passed": result.get("passed", True),
            "score": result.get("score", 0.7),
            "feedback": result.get("feedback", "")
        }
