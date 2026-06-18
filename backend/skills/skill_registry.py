"""Skill注册表 - 管理所有可用Skill"""
import logging
from typing import Dict, Optional
from backend.skills.base_skill import BaseSkill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Skill注册表 - 动态加载和管理"""

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        """注册Skill"""
        self._skills[skill.name] = skill
        logger.info(f"[Registry] Registered skill: {skill.name}")

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """获取Skill实例"""
        return self._skills.get(name)

    def list_skills(self):
        """列出所有已注册Skill"""
        return {name: skill.description for name, skill in self._skills.items()}

    def get_all_stats(self):
        """获取所有Skill统计"""
        return {name: skill.get_stats() for name, skill in self._skills.items()}


# 全局注册表
skill_registry = SkillRegistry()


def register_all_skills():
    """注册所有Skill"""
    from backend.skills.jd_parser_skill import JDParserSkill
    from backend.skills.resume_generator_skill import ResumeGeneratorSkill
    from backend.skills.data_preprocessing_skill import DataPreprocessingSkill
    from backend.skills.database_operation_skill import DatabaseOperationSkill
    from backend.skills.rag_retrieval_skill import RAGRetrievalSkill
    from backend.skills.matching_evaluation_skill import MatchingEvaluationSkill
    from backend.skills.shap_explainer_skill import SHAPExplainerSkill
    from backend.skills.feedback_learning_skill import FeedbackLearningSkill

    skill_registry.register(JDParserSkill())
    skill_registry.register(ResumeGeneratorSkill())
    skill_registry.register(DataPreprocessingSkill())
    skill_registry.register(DatabaseOperationSkill())
    skill_registry.register(RAGRetrievalSkill())
    skill_registry.register(MatchingEvaluationSkill())
    skill_registry.register(SHAPExplainerSkill())
    skill_registry.register(FeedbackLearningSkill())
    logger.info(f"[Registry] All {len(skill_registry._skills)} skills registered")
