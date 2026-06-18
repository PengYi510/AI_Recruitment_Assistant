"""Harness全局控制器 - 生成-评估分离架构+动态调度"""
import logging, time
from typing import Dict, Any, List, Optional
from backend.harness.state import TaskState, TaskStatus
from backend.harness.dynamic_scheduler import dynamic_scheduler
from backend.agents.planner_agent import PlannerAgent
from backend.agents.generator_agent import GeneratorAgent
from backend.agents.evaluator_agent import EvaluatorAgent
from backend.database.models import hr_db

logger = logging.getLogger(__name__)


class HarnessController:
    """Harness全局控制器
    核心职责:
    1. 接收用户请求，创建执行计划
    2. 协调Planner-Generator-Evaluator三Agent
    3. 动态调度任务迭代
    4. 权限管理+审计日志+错误恢复
    5. 状态持久化+历史反馈学习
    """
    def __init__(self):
        self.planner = PlannerAgent()
        self.generator = GeneratorAgent()
        self.evaluator = EvaluatorAgent()

    async def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行完整的Harness流程"""
        start_time = time.time()
        context = context or {}

        # 1. 创建执行计划
        state = dynamic_scheduler.create_execution_plan(query)
        logger.info(f"[Harness] Task {state.task_id}: complexity={state.complexity_score:.2f}, "
                   f"max_iter={state.max_iterations}, subtasks={len(state.sub_tasks)}")

        try:
            # 2. Planner规划阶段
            state.status = TaskStatus.PLANNING
            plan = await self.planner.plan(query, state, context)
            state.plan = plan
            state.add_history("planning_complete", str(plan.get("task_type", "")))

            # 2.5 对于 candidate_search 类型，不分解子任务（保持完整查询的所有约束）
            # 子任务分解会导致硬约束（985/应届生/全日制等）在子任务间丢失
            if plan.get("task_type") == "candidate_search" and len(state.sub_tasks) > 1:
                logger.info(f"[Harness] candidate_search: merging {len(state.sub_tasks)} subtasks into 1")
                merged_subtask = state.sub_tasks[0]
                merged_subtask.description = query  # 使用完整原始查询
                state.sub_tasks = [merged_subtask]

            # 3. 对每个子任务执行生成-评估循环
            results = []
            for subtask in state.sub_tasks:
                subtask_result = await self._execute_subtask(subtask, state, context)
                results.append(subtask_result)

            # 4. 聚合结果
            state.final_result = self._aggregate_results(results, state)
            state.mark_completed()

            # 5. 记录反馈
            response_time = time.time() - start_time
            dynamic_scheduler.record_feedback(
                complexity=state.complexity_score,
                success=True,
                iterations=state.current_iteration,
                response_time=response_time
            )
            hr_db.save_system_feedback(
                task_type=plan.get("task_type", "general"),
                complexity_score=state.complexity_score,
                iterations=state.current_iteration,
                success=True,
                response_time=response_time
            )

            return {
                "success": True,
                "result": state.final_result,
                "task_id": state.task_id,
                "complexity": state.complexity_score,
                "iterations": state.current_iteration,
                "response_time": round(response_time, 2),
            }

        except Exception as e:
            logger.error(f"[Harness] Task {state.task_id} failed: {e}")
            state.mark_failed(str(e))
            response_time = time.time() - start_time
            dynamic_scheduler.record_feedback(
                complexity=state.complexity_score,
                success=False,
                iterations=state.current_iteration,
                response_time=response_time
            )
            return {
                "success": False,
                "error": str(e),
                "task_id": state.task_id,
                "complexity": state.complexity_score,
            }

    async def _execute_subtask(self, subtask, state: TaskState, context: Dict[str, Any]) -> Any:
        """执行单个子任务的生成-评估循环"""
        for iteration in range(subtask.max_iterations):
            state.current_iteration = iteration + 1
            subtask.iterations = iteration + 1

            # 生成阶段
            state.status = TaskStatus.GENERATING
            subtask.status = TaskStatus.GENERATING
            generation = await self.generator.generate(subtask, state, context)

            # 评估阶段
            state.status = TaskStatus.EVALUATING
            subtask.status = TaskStatus.EVALUATING
            evaluation = await self.evaluator.evaluate(generation, subtask, state)

            if evaluation.get("passed", False):
                subtask.status = TaskStatus.COMPLETED
                subtask.result = generation
                return generation

            # 不通过则生成返工指令
            if iteration < subtask.max_iterations - 1:
                state.status = TaskStatus.REWORKING
                context["rework_feedback"] = evaluation.get("feedback", "")
                logger.info(f"[Harness] Subtask {subtask.task_id} rework iteration {iteration+1}")

        # 达到最大迭代次数，返回最后一次结果
        subtask.status = TaskStatus.COMPLETED
        subtask.result = generation
        return generation

    def _aggregate_results(self, results: List[Any], state: TaskState) -> Dict[str, Any]:
        """聚合多个子任务结果"""
        if len(results) == 1:
            return results[0] if isinstance(results[0], dict) else {"result": results[0]}
        return {"results": results, "subtask_count": len(results)}


# 全局实例
harness = HarnessController()
