"""测试Skill注册表"""
import pytest


class TestSkillRegistry:
    def test_registry_import(self):
        from backend.skills.skill_registry import SkillRegistry
        registry = SkillRegistry()
        assert registry is not None

    def test_register_and_get(self):
        from backend.skills.skill_registry import SkillRegistry
        from backend.skills.base_skill import BaseSkill
        class DummySkill(BaseSkill):
            def __init__(self):
                super().__init__(name="dummy", description="dummy")
            async def execute(self, params):
                return {}
        registry = SkillRegistry()
        registry.register(DummySkill())
        assert registry.get_skill("dummy") is not None
        assert registry.get_skill("nonexist") is None

    def test_list_skills(self):
        from backend.skills.skill_registry import SkillRegistry
        from backend.skills.base_skill import BaseSkill
        class S1(BaseSkill):
            def __init__(self): super().__init__(name="s1", description="s1")
            async def execute(self, p): return {}
        class S2(BaseSkill):
            def __init__(self): super().__init__(name="s2", description="s2")
            async def execute(self, p): return {}
        registry = SkillRegistry()
        registry.register(S1())
        registry.register(S2())
        names = registry.list_skills()
        assert "s1" in names
        assert "s2" in names

    def test_get_all_stats(self):
        from backend.skills.skill_registry import SkillRegistry
        from backend.skills.base_skill import BaseSkill
        class TestSkill(BaseSkill):
            def __init__(self): super().__init__(name="test", description="test")
            async def execute(self, p): return {}
        registry = SkillRegistry()
        registry.register(TestSkill())
        stats = registry.get_all_stats()
        assert "test" in stats

    def test_register_all_skills(self):
        """测试 register_all_skills() 注册所有8个Skill"""
        from backend.skills.skill_registry import skill_registry, register_all_skills
        # 先清空
        skill_registry._skills = {}
        register_all_skills()
        skills = skill_registry.list_skills()
        assert len(skills) == 8
        expected_names = [
            "jd_parser", "resume_generator", "data_preprocessing",
            "database_operation", "rag_retrieval", "matching_evaluation",
            "shap_explainer", "feedback_learning"
        ]
        for name in expected_names:
            assert name in skills, f"Skill '{name}' not registered"

    def test_register_all_skills_idempotent(self):
        """测试 register_all_skills() 重复调用不会报错"""
        from backend.skills.skill_registry import skill_registry, register_all_skills
        skill_registry._skills = {}
        register_all_skills()
        register_all_skills()  # 再次调用不应报错
        assert len(skill_registry.list_skills()) == 8

    def test_register_duplicate_overwrites(self):
        """测试重复注册同名 Skill 会覆盖"""
        from backend.skills.skill_registry import SkillRegistry
        from backend.skills.base_skill import BaseSkill
        class SkillV1(BaseSkill):
            def __init__(self): super().__init__(name="dup", description="v1")
            async def execute(self, p): return {}
        class SkillV2(BaseSkill):
            def __init__(self): super().__init__(name="dup", description="v2")
            async def execute(self, p): return {}
        registry = SkillRegistry()
        registry.register(SkillV1())
        registry.register(SkillV2())
        skill = registry.get_skill("dup")
        assert skill.description == "v2"
