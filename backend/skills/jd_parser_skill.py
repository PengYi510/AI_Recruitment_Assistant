"""JD解析Skill - 支持短文本查询和长文本JD解析"""
import logging
from typing import Dict, Any
from backend.skills.base_skill import BaseSkill
from backend.models.longcat_client import chat_json

logger = logging.getLogger(__name__)


class JDParserSkill(BaseSkill):
    """JD解析Skill: 提取硬性规则和软性要求"""

    def __init__(self):
        super().__init__(name="jd_parser", description="解析JD/查询,提取硬性规则和软性要求")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        system = """你是资深HR需求分析师。解析招聘需求,提取结构化信息。
返回JSON:
{
  "hard_rules": [{"field": "字段名", "operator": "=|>=|<=|!=|contains|not_contains", "value": "值"}],
  "soft_requirements": [{"field": "字段名", "importance": "high|medium|low", "description": "描述"}],
  "education_req": "学历要求",
  "min_experience": 0,
  "max_salary": 0,
  "required_skills": ["技能1"],
  "preferred_skills": ["技能2"],
  "location": "工作地点",
  "industry": "行业",
  "is_management": false,
  "summary": "需求概述"
}"""
        try:
            result = chat_json(system=system, user=f"解析需求: {query}")
            logger.info(f"[JDParser] Parsed: {len(result.get('hard_rules', []))} hard rules, "
                       f"{len(result.get('soft_requirements', []))} soft reqs")
            return result
        except Exception as e:
            logger.error(f"[JDParser] Parse failed: {e}")
            return {"hard_rules": [], "soft_requirements": [], "summary": query,
                    "required_skills": [], "education_req": "", "min_experience": 0}
