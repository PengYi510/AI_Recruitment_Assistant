"""测试CatBoost匹配器"""
import pytest
import numpy as np


class TestCatBoostMatcher:
    def test_extract_structured_features(self):
        from backend.models.catboost_matcher import catboost_matcher
        jd = {"min_experience": 3, "education_req": "硕士",
              "required_skills": ["Python", "Java"], "preferred_skills": ["Go"]}
        candidate = {
            "work_years": 5, "highest_education": "硕士",
            "skills": [{"skill_name": "Python"}, {"skill_name": "Java"}, {"skill_name": "Go"}],
            "work_experiences": [{"company_name": "美团"}],
            "projects": [{"project_name": "推荐系统"}],
        }
        features = catboost_matcher.extract_structured_features(jd, candidate)
        assert features.shape == (12,)
        assert all(0 <= f <= 1 for f in features)

    def test_predict(self):
        from backend.models.catboost_matcher import catboost_matcher
        features = np.random.rand(12)
        score = catboost_matcher.predict(features)
        assert 0 <= score <= 1

    def test_get_feature_importance(self):
        from backend.models.catboost_matcher import catboost_matcher
        importance = catboost_matcher.get_feature_importance()
        assert len(importance) == 12
        assert all(v >= 0 for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_batch_predict(self):
        from backend.models.catboost_matcher import catboost_matcher
        features_batch = np.random.rand(10, 12)
        scores = [catboost_matcher.predict(f) for f in features_batch]
        assert len(scores) == 10
        assert all(0 <= s <= 1 for s in scores)

    def test_update_weights(self):
        from backend.models.catboost_matcher import catboost_matcher
        # 不应该抛错
        catboost_matcher.update_weights({"adjustment": "test"})
