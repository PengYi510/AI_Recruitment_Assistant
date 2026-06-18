"""测试数据库模型"""
import pytest
from backend.database.models import HRDatabase


@pytest.fixture
def db(tmp_path):
    db_file = str(tmp_path / "test.db")
    return HRDatabase(db_path=db_file)


class TestHRDatabase:
    def test_insert_candidate(self, db, sample_candidate):
        cid = db.insert_candidate(sample_candidate)
        assert cid > 0

    def test_get_candidate(self, db, sample_candidate):
        cid = db.insert_candidate(sample_candidate)
        result = db.get_candidate(cid)
        assert result is not None
        assert result["name"] == "张三"
        assert result["highest_education"] == "硕士"
        assert "skills" in result
        assert "work_experiences" in result
        assert "projects" in result
        assert "awards_certificates" in result

    def test_insert_skill(self, db, sample_candidate):
        cid = db.insert_candidate(sample_candidate)
        sid = db.insert_skill(cid, "Python", 5)
        assert sid > 0
        cand = db.get_candidate(cid)
        assert len(cand["skills"]) == 1
        assert cand["skills"][0]["skill_name"] == "Python"

    def test_insert_work_experience(self, db, sample_candidate, sample_work_exp):
        cid = db.insert_candidate(sample_candidate)
        wid = db.insert_work_experience(cid, sample_work_exp)
        assert wid > 0
        cand = db.get_candidate(cid)
        assert len(cand["work_experiences"]) == 1
        assert cand["work_experiences"][0]["company_name"] == "美团"

    def test_insert_project(self, db, sample_candidate, sample_project):
        cid = db.insert_candidate(sample_candidate)
        pid = db.insert_project(cid, sample_project)
        assert pid > 0
        cand = db.get_candidate(cid)
        assert len(cand["projects"]) == 1

    def test_insert_award_certificate(self, db, sample_candidate):
        cid = db.insert_candidate(sample_candidate)
        aid = db.insert_award_certificate(cid, {
            "name": "PMP认证",
            "type": "资格证书",
            "date": "2023-06",
            "description": "项目管理专业认证"
        })
        assert aid > 0
        cand = db.get_candidate(cid)
        assert len(cand["awards_certificates"]) == 1

    def test_search_candidates(self, db, sample_candidate):
        for i in range(10):
            c = sample_candidate.copy()
            c["name"] = f"候选人_{i}"
            db.insert_candidate(c)
        results = db.search_candidates(limit=5)
        assert len(results) == 5

    def test_search_with_filters(self, db, sample_candidate):
        db.insert_candidate(sample_candidate)
        c2 = sample_candidate.copy()
        c2["highest_education"] = "本科"
        c2["name"] = "李四"
        db.insert_candidate(c2)
        results = db.search_candidates(filters={"highest_education": "硕士"})
        assert all(r["highest_education"] == "硕士" for r in results)

    def test_count(self, db, sample_candidate):
        assert db.get_all_candidates_count() == 0
        db.insert_candidate(sample_candidate)
        assert db.get_all_candidates_count() == 1

    def test_matching_history(self, db):
        hid = db.insert_matching_history("Python工程师", [1, 2, 3], [0.9, 0.8, 0.7], 150.5)
        assert hid > 0
        db.update_feedback(hid, 1)
        feedback = db.get_recent_feedback(10)
        assert len(feedback) == 1
        assert feedback[0]["success"] == 1

    def test_performance_stats(self, db):
        db.insert_matching_history("test", [1], [0.9], 100)
        db.update_feedback(1, 1)
        stats = db.get_performance_stats()
        assert stats["total_queries"] == 1
        assert stats["positive_feedback"] == 1

    def test_get_nonexistent_candidate(self, db):
        result = db.get_candidate(9999)
        assert result is None
