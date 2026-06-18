"""Planner Agent - 任务规划与分解"""
import logging
import re
from typing import Dict, Any
from backend.models.longcat_client import chat_json
from backend.harness.state import TaskState

logger = logging.getLogger(__name__)


class PlannerAgent:
    """规划Agent: 解析用户意图,制定执行计划和动态验收标准"""

    # JD长文本检测阈值
    _JD_LENGTH_THRESHOLD = 100

    async def plan(self, query: str, state: TaskState, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """制定任务执行计划

        增强逻辑：
        - 当检测到输入是完整JD/岗位描述时，强制task_type为candidate_search
        - 避免LLM误判为jd_parse（解析JD本身而非搜索候选人）
        """
        # 前置判断：如果是完整JD文本，直接走candidate_search
        if self._is_full_jd(query):
            logger.info("[Planner] Detected full JD input, forcing candidate_search")
            return {
                "task_type": "candidate_search",
                "skills_needed": ["rag_retrieval", "matching_evaluation", "shap_explainer"],
                "acceptance_criteria": "根据JD要求检索并返回匹配的候选人列表，含SHAP可解释性",
                "priority": "high",
                "requires_multimodal": False
            }

        try:
            system = """你是HR智能匹配系统的任务规划专家。分析用户需求,制定执行计划。

重要规则：
- 当用户输入包含岗位描述、职责要求、任职要求等内容时，task_type 必须为 "candidate_search"（目的是搜索匹配候选人）
- 只有当用户明确要求"解析JD"、"提取JD信息"时才用 "jd_parse"
- 当用户要求推荐/搜索/匹配候选人时，task_type 为 "candidate_search"

返回JSON:
{
  "task_type": "jd_parse|candidate_search|matching|explanation|feedback|data_generate",
  "skills_needed": ["skill1", "skill2"],
  "acceptance_criteria": "验收标准描述",
  "priority": "high|medium|low",
  "requires_multimodal": true/false
}"""
            plan = chat_json(system=system, user=f"规划任务: {query}")
            logger.info(f"[Planner] Plan: type={plan.get('task_type')}, skills={plan.get('skills_needed')}")

            # 后置校正：即使LLM返回jd_parse，如果query明显是要搜索候选人，修正为candidate_search
            if plan.get("task_type") == "jd_parse" and self._should_search_candidates(query):
                logger.info("[Planner] Post-correction: jd_parse -> candidate_search (user likely wants to search)")
                plan["task_type"] = "candidate_search"
                plan["skills_needed"] = ["rag_retrieval", "matching_evaluation", "shap_explainer"]

            return plan
        except Exception as e:
            logger.warning(f"[Planner] Planning failed: {e}, using default plan")
            return {
                "task_type": "candidate_search",
                "skills_needed": ["rag_retrieval", "matching_evaluation"],
                "acceptance_criteria": "返回相关候选人列表",
                "priority": "medium",
                "requires_multimodal": False
            }

    def _is_full_jd(self, query: str) -> bool:
        """判断输入是否为完整的JD/岗位描述文本

        特征：
        - 长度超过阈值
        - 包含岗位描述的典型结构（职责、要求、学历等关键词）
        """
        if len(query) < self._JD_LENGTH_THRESHOLD:
            return False

        # 检测JD结构性关键词（至少命中2个）
        jd_indicators = [
            r'岗位职责|工作职责|职责描述',
            r'岗位要求|任职要求|基本要求|职位要求',
            r'岗位亮点|福利待遇|薪资',
            r'本科|硕士|博士|学历要求',
            r'\d{4}届|实习|应届',
            r'部门介绍|团队介绍|公司介绍',
            r'熟练掌握|熟悉|精通',
            r'编程能力|开发经验|项目经验',
        ]
        hit_count = sum(1 for pattern in jd_indicators if re.search(pattern, query))
        return hit_count >= 2

    def _should_search_candidates(self, query: str) -> bool:
        """判断用户意图是否为搜索候选人（即使LLM判断为jd_parse）

        当query包含以下特征时，应走candidate_search：
        - 包含招聘/推荐/搜索等动作词
        - 包含具体的技能要求+学历要求（暗示要找人）
        - query长度较长且包含岗位结构
        """
        # 包含明确的搜索意图词
        search_intent = re.search(r'帮我找|推荐|搜索|匹配|筛选|招聘|招人', query)
        if search_intent:
            return True

        # 长文本+包含岗位结构 → 暗示用户要用这个JD搜人
        if len(query) > self._JD_LENGTH_THRESHOLD:
            has_requirements = bool(re.search(r'岗位要求|任职要求|基本要求', query))
            has_skills = bool(re.search(r'Python|Java|SQL|机器学习|数据', query))
            if has_requirements and has_skills:
                return True

        return False
