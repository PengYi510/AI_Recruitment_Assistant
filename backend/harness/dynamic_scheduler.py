"""反馈驱动动态任务调度算法 - 核心创新点1

算法伪代码:
  Input: user_query, feedback_history
  Output: execution_plan

  1. complexity = LLM_evaluate(user_query)  # O(1)
  2. threshold = adjust_threshold(feedback_history)  # O(n), n=feedback_size
  3. if complexity < threshold_simple:
       iterations = 1, subtasks = [query]
     elif complexity < threshold_complex:
       iterations = 2, subtasks = split(query, 2)
     else:
       iterations = 3, subtasks = split(query, 3)
  4. for subtask in subtasks:
       result = generate(subtask)
       evaluation = evaluate(result)
       if not evaluation.pass:
         result = regenerate(subtask, evaluation.feedback)
  5. return aggregate(results)

时间复杂度: O(n + k*m)
  n = feedback_history_size (阈值调整)
  k = subtask_count (最多3)
  m = max_iterations (最多3)
"""
import logging, time, uuid
from typing import Dict, Any, List, Optional, Tuple
from backend.config import (COMPLEXITY_SIMPLE_THRESHOLD, COMPLEXITY_COMPLEX_THRESHOLD,
    MAX_ITERATIONS_SIMPLE, MAX_ITERATIONS_MEDIUM, MAX_ITERATIONS_COMPLEX,
    MAX_SUBTASKS, FEEDBACK_HISTORY_SIZE)
from backend.harness.state import TaskState, SubTask, TaskStatus
from backend.models.longcat_client import chat_json

logger = logging.getLogger(__name__)


class DynamicScheduler:
    """反馈驱动的动态任务调度器"""

    def __init__(self):
        self.simple_threshold = COMPLEXITY_SIMPLE_THRESHOLD
        self.complex_threshold = COMPLEXITY_COMPLEX_THRESHOLD
        self.feedback_history: List[Dict[str, Any]] = []
        self._adjustment_count = 0

    def evaluate_complexity(self, query: str) -> float:
        """评估任务复杂度 (0-1)"""
        try:
            system = """你是任务复杂度评估专家。根据用户查询评估任务复杂度。
返回JSON: {"complexity": 0.0-1.0, "reason": "原因"}
评分标准:
- 简单(<0.3): 单一条件查询、直接信息检索
- 中等(0.3-0.7): 多条件筛选、需要分析对比
- 复杂(>0.7): 多维度匹配、需要深度分析和解释"""
            result = chat_json(system=system, user=f"评估查询复杂度: {query}")
            score = float(result.get("complexity", 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Complexity evaluation failed: {e}, using heuristic")
            return self._heuristic_complexity(query)

    def _heuristic_complexity(self, query: str) -> float:
        """启发式复杂度评估（降级方案）"""
        score = 0.3
        if len(query) > 100:
            score += 0.2
        keywords_complex = ["对比", "分析", "排名", "推荐", "匹配", "解释", "详细"]
        keywords_simple = ["查找", "查询", "搜索", "列出"]
        for kw in keywords_complex:
            if kw in query:
                score += 0.1
        for kw in keywords_simple:
            if kw in query:
                score -= 0.05
        return max(0.0, min(1.0, score))

    def determine_iterations(self, complexity: float) -> int:
        """根据复杂度确定迭代次数"""
        if complexity < self.simple_threshold:
            return MAX_ITERATIONS_SIMPLE
        elif complexity < self.complex_threshold:
            return MAX_ITERATIONS_MEDIUM
        else:
            return MAX_ITERATIONS_COMPLEX

    def split_subtasks(self, query: str, complexity: float) -> List[SubTask]:
        """将复杂任务拆分为原子子任务（最多3个）"""
        if complexity < self.complex_threshold:
            return [SubTask(task_id=str(uuid.uuid4())[:8], description=query, complexity=complexity)]

        try:
            system = """你是任务分解专家。将复杂查询分解为不超过3个原子子任务。
返回JSON: {"subtasks": [{"description": "子任务描述"}]}"""
            result = chat_json(system=system, user=f"分解任务: {query}")
            subtasks = []
            for i, st in enumerate(result.get("subtasks", [{"description": query}])[:MAX_SUBTASKS]):
                subtasks.append(SubTask(
                    task_id=str(uuid.uuid4())[:8],
                    description=st["description"],
                    complexity=complexity / len(result.get("subtasks", [1]))
                ))
            return subtasks if subtasks else [SubTask(task_id=str(uuid.uuid4())[:8], description=query, complexity=complexity)]
        except Exception:
            return [SubTask(task_id=str(uuid.uuid4())[:8], description=query, complexity=complexity)]

    def adjust_thresholds(self):
        """基于历史反馈动态调整复杂度评分阈值"""
        if len(self.feedback_history) < 10:
            return
        recent = self.feedback_history[-FEEDBACK_HISTORY_SIZE:]
        success_by_complexity = {"simple": [], "medium": [], "complex": []}
        for fb in recent:
            c = fb.get("complexity", 0.5)
            s = fb.get("success", False)
            if c < self.simple_threshold:
                success_by_complexity["simple"].append(s)
            elif c < self.complex_threshold:
                success_by_complexity["medium"].append(s)
            else:
                success_by_complexity["complex"].append(s)

        # 如果简单任务失败率过高，降低阈值（更多任务归为中等）
        if success_by_complexity["simple"]:
            simple_rate = sum(success_by_complexity["simple"]) / len(success_by_complexity["simple"])
            if simple_rate < 0.8:
                self.simple_threshold = max(0.2, self.simple_threshold - 0.05)
        # 如果复杂任务成功率高，提高阈值（减少不必要的复杂处理）
        if success_by_complexity["complex"]:
            complex_rate = sum(success_by_complexity["complex"]) / len(success_by_complexity["complex"])
            if complex_rate > 0.9:
                self.complex_threshold = min(0.8, self.complex_threshold + 0.05)

        self._adjustment_count += 1
        logger.info(f"Thresholds adjusted (#{self._adjustment_count}): "
                   f"simple={self.simple_threshold:.2f}, complex={self.complex_threshold:.2f}")

    def record_feedback(self, complexity: float, success: bool, iterations: int, response_time: float):
        """记录任务执行反馈（系统自动判定）"""
        self.feedback_history.append({
            "complexity": complexity, "success": success,
            "iterations": iterations, "response_time": response_time,
            "timestamp": time.time(), "source": "system"
        })
        if len(self.feedback_history) > FEEDBACK_HISTORY_SIZE * 2:
            self.feedback_history = self.feedback_history[-FEEDBACK_HISTORY_SIZE:]
        self.adjust_thresholds()

    def record_user_feedback(self, satisfied: bool):
        """记录用户满意度反馈（来自前端点赞/点踩），驱动阈值动态调整。

        与 record_feedback 的区别：
        - record_feedback: 系统自动判定执行是否成功（内部调用）
        - record_user_feedback: 用户主观评价结果质量（外部反馈驱动）

        调整策略：
        - 用户不满意 → 可能是简单任务处理得不够好 → 降低 simple_threshold
          使更多任务获得更多迭代次数，提升处理质量
        - 用户持续满意 → 当前阈值合理或偏保守 → 微调提升效率
        """
        if not hasattr(self, '_user_feedback_history'):
            self._user_feedback_history = []

        self._user_feedback_history.append({
            "satisfied": satisfied,
            "timestamp": time.time()
        })

        # 只保留最近的反馈
        if len(self._user_feedback_history) > FEEDBACK_HISTORY_SIZE:
            self._user_feedback_history = self._user_feedback_history[-FEEDBACK_HISTORY_SIZE:]

        # 至少 5 条用户反馈才触发阈值调整
        if len(self._user_feedback_history) < 5:
            return

        recent = self._user_feedback_history[-20:]
        satisfaction_rate = sum(1 for f in recent if f["satisfied"]) / len(recent)

        # 用户满意率低于 60% → 降低 simple_threshold（让更多任务获得多轮迭代）
        if satisfaction_rate < 0.6:
            old_threshold = self.simple_threshold
            self.simple_threshold = max(0.15, self.simple_threshold - 0.03)
            logger.info(f"[UserFeedback] satisfaction_rate={satisfaction_rate:.2f} < 0.6, "
                       f"simple_threshold: {old_threshold:.2f} → {self.simple_threshold:.2f}")
        # 用户满意率高于 85% → 微调提高效率（提高 simple_threshold）
        elif satisfaction_rate > 0.85:
            old_threshold = self.simple_threshold
            self.simple_threshold = min(0.4, self.simple_threshold + 0.02)
            logger.info(f"[UserFeedback] satisfaction_rate={satisfaction_rate:.2f} > 0.85, "
                       f"simple_threshold: {old_threshold:.2f} → {self.simple_threshold:.2f}")

        self._adjustment_count += 1

    def create_execution_plan(self, query: str) -> TaskState:
        """创建完整执行计划"""
        task_id = str(uuid.uuid4())[:8]
        state = TaskState(task_id=task_id, user_query=query, start_time=time.time())

        # 评估复杂度
        complexity = self.evaluate_complexity(query)
        state.complexity_score = complexity

        # 确定迭代次数
        state.max_iterations = self.determine_iterations(complexity)

        # 拆分子任务
        state.sub_tasks = self.split_subtasks(query, complexity)
        for st in state.sub_tasks:
            st.max_iterations = state.max_iterations

        state.status = TaskStatus.PLANNING
        state.add_history("plan_created",
            f"complexity={complexity:.2f}, iterations={state.max_iterations}, "
            f"subtasks={len(state.sub_tasks)}")

        return state

    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息（含用户反馈维度）"""
        if not self.feedback_history:
            system_stats = {"total_tasks": 0, "success_rate": 0, "avg_response_time": 0}
        else:
            total = len(self.feedback_history)
            successes = sum(1 for f in self.feedback_history if f["success"])
            avg_time = sum(f["response_time"] for f in self.feedback_history) / total
            system_stats = {
                "total_tasks": total,
                "success_rate": round(successes / total, 4),
                "avg_response_time": round(avg_time, 2),
            }

        # 用户反馈统计
        user_fb = getattr(self, '_user_feedback_history', [])
        if user_fb:
            user_satisfied = sum(1 for f in user_fb if f["satisfied"])
            user_satisfaction_rate = round(user_satisfied / len(user_fb), 4)
        else:
            user_satisfaction_rate = None

        return {
            **system_stats,
            "simple_threshold": self.simple_threshold,
            "complex_threshold": self.complex_threshold,
            "adjustment_count": self._adjustment_count,
            "user_feedback_count": len(user_fb),
            "user_satisfaction_rate": user_satisfaction_rate,
        }


# 全局实例
dynamic_scheduler = DynamicScheduler()
