"""测试配置和共享fixtures"""
import sys, os, pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock环境变量
os.environ.setdefault("LLM_API_KEY", "test-key-12345")
os.environ.setdefault("LLM_BASE_URL", "https://test.api.com/v1")
os.environ.setdefault("DB_PATH", ":memory:")


@pytest.fixture(autouse=True)
def mock_db_path(tmp_path, monkeypatch):
    """使用临时目录的数据库"""
    db_file = str(tmp_path / "test_hr.db")
    monkeypatch.setattr("backend.config.DB_PATH", db_file)
    return db_file


@pytest.fixture
def sample_candidate():
    return {
        "name": "张三",
        "gender": "男",
        "age": 28,
        "birth_date": "1997-03-15",
        "phone": "13800138000",
        "email": "zhangsan@test.com",
        "address": "北京市海淀区",
        "highest_education": "硕士",
        "work_years": 5,
        "current_salary": 35000,
        "expected_salary": 45000,
        "job_status": "在职看机会",
        "current_position": "高级工程师",
        "location": "北京",
        "summary": "硕士学历,5年工作经验",
    }


@pytest.fixture
def sample_jd_info():
    return {
        "hard_rules": [
            {"field": "highest_education", "operator": "=", "value": "硕士"},
            {"field": "work_years", "operator": ">=", "value": "3"},
        ],
        "soft_requirements": [
            {"field": "skills", "importance": "high", "description": "熟悉Python"},
        ],
        "education_req": "硕士",
        "min_experience": 3,
        "required_skills": ["Python", "Java"],
        "preferred_skills": ["Go", "Kubernetes"],
        "summary": "高级后端工程师"
    }


@pytest.fixture
def sample_work_exp():
    return {
        "company_name": "美团",
        "position": "高级工程师",
        "start_date": "2020-07-01",
        "end_date": "2024-01-01",
        "description": "负责核心系统开发",
    }


@pytest.fixture
def sample_project():
    return {
        "project_name": "智能推荐系统",
        "role": "技术负责人",
        "start_date": "2022-01-01",
        "end_date": "2023-06-01",
        "description": "设计推荐引擎",
        "technologies": "Python,TensorFlow,Redis",
    }
