"""全局配置模块 - Harness驱动的多模态分层融合智能招聘匹配系统"""

import os
from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_DIR = DATA_DIR / "sqlite"
CHROMA_DIR = DATA_DIR / "chroma"
SHAP_DIR = DATA_DIR / "shap"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

for d in [SQLITE_DIR, CHROMA_DIR, SHAP_DIR, SYNTHETIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM 配置（保留原有LongCat API调用方式） ──────────────────────────────────
LLM_API_KEY = os.environ.get("AIBP_API_KEY", "1688758691006349378")
LLM_BASE_URL = "https://aigc.sankuai.com/v1/openai/native"
LLM_MODEL = "LongCat-Flash-Chat"
LLM_RATE_LIMIT_RETRIES = 3
LLM_RATE_LIMIT_WAIT = 20

# ── 数据库配置 ──────────────────────────────────────────────────────────────
SQLITE_DB_PATH = str(SQLITE_DIR / "hr_matching.db")
DB_PATH = SQLITE_DB_PATH  # 别名
CONVERSATION_DB_PATH = str(DATA_DIR / "conversation_sessions.db")

# ── 向量库配置 ──────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "candidates_collection"
CHROMA_COLLECTION = CHROMA_COLLECTION_NAME  # 别名
CHROMA_PERSIST_DIR = CHROMA_DIR  # Path对象
EMBEDDING_DIMENSION = 1024

# ── Embedding 模型配置 ─────────────────────────────────────────────────────────
# 使用 BAAI/bge-m3 模型（1024维，567M参数，~2.2GB）
# 首次使用需从 HuggingFace 下载，放置于 models/bge-m3/ 目录
# 下载方式见 README.md 或 how_to_use.md
EMBEDDING_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-m3")

# ── Harness 动态调度配置 ─────────────────────────────────────────────────────
COMPLEXITY_THRESHOLD = 0.7  # 动态阈值初始值
COMPLEXITY_SIMPLE_THRESHOLD = 0.3
COMPLEXITY_COMPLEX_THRESHOLD = 0.7
HARNESS_MAX_ITERATIONS = 3
MAX_ITERATIONS_SIMPLE = 1
MAX_ITERATIONS_MEDIUM = 2
MAX_ITERATIONS_COMPLEX = 3
MAX_SUBTASKS = 3
FEEDBACK_HISTORY_SIZE = 100

# ── 匹配模型配置 ─────────────────────────────────────────────────────────────
TEXT_EMBEDDING_DIM = 1024
IMAGE_EMBEDDING_DIM = 768
FUSION_EMBEDDING_DIM = 1024
STRUCTURED_FEATURE_DIM = 12
MULTIMODAL_WEIGHT = 0.6
STRUCTURED_WEIGHT = 0.4

# ── RAG 检索配置 ──────────────────────────────────────────────────────────────
RAG_TOP_K = 20
# 两路召回融合权重：BM25 稀疏 0.3 + BGE-M3 稠密 0.7（系统初始设计值，
# 稠密语义检索为主、稀疏关键词检索为辅）。融合前对两路分数各做 min-max 归一化。
RAG_BM25_WEIGHT = 0.3
RAG_DENSE_WEIGHT = 0.7
FINAL_TOP_K = 10

# ── SHAP 可解释性配置 ─────────────────────────────────────────────────────────
# 完整12维基础特征名（英文key，内部计算使用）
SHAP_FEATURE_KEYS = [
    "education_match", "skill_match", "experience_match",
    "salary_match", "location_match", "industry_match",
    "project_relevance", "certification_match", "language_match",
    "management_match", "cultural_fit", "growth_potential"
]

# 中文特征名映射（用于图表展示）
SHAP_FEATURE_NAMES_CN = {
    "education_match": "学历匹配度",
    "skill_match": "技能匹配度",
    "experience_match": "工作经验匹配度",
    "salary_match": "薪资期望匹配度",
    "location_match": "地域匹配度",
    "industry_match": "行业匹配度",
    "project_relevance": "项目相关度",
    "certification_match": "证书匹配度",
    "language_match": "语言能力匹配度",
    "management_match": "管理经验匹配度",
    "cultural_fit": "文化契合度",
    "growth_potential": "成长潜力",
}

# ── 动态特征注册表（LLM 提取的 extra_constraints 自动映射为 SHAP 特征）────────
# 当 LLM 从查询中提取到 extra_constraints 时，系统会自动为每个约束生成对应的
# SHAP 特征维度，无需手动添加代码。以下是已知动态特征的元数据注册表。
# 新属性只需在此处注册即可自动获得 SHAP 解释能力；未注册的属性也能工作，
# 只是使用默认的中文名和权重。
DYNAMIC_FEATURE_REGISTRY = {
    "gpa": {
        "cn_name": "GPA匹配度",
        "weight": 0.10,
        "baseline": 0.50,
        "description": "候选人GPA与查询要求的匹配程度",
    },
    "target_job": {
        "cn_name": "目标岗位匹配度",
        "weight": 0.12,
        "baseline": 0.30,
        "description": "候选人求职意向与查询岗位的匹配程度",
    },
    "hobbies": {
        "cn_name": "兴趣爱好匹配度",
        "weight": 0.03,
        "baseline": 0.40,
        "description": "候选人兴趣爱好与查询要求的匹配程度",
    },
    "languages": {
        "cn_name": "语言能力匹配度",
        "weight": 0.06,
        "baseline": 0.45,
        "description": "候选人语言能力与查询要求的匹配程度",
    },
    "height": {
        "cn_name": "身高匹配度",
        "weight": 0.02,
        "baseline": 0.50,
        "description": "候选人身高与查询要求的匹配程度",
    },
    "weight": {
        "cn_name": "体重匹配度",
        "weight": 0.02,
        "baseline": 0.50,
        "description": "候选人体重与查询要求的匹配程度",
    },
    "ethnicity": {
        "cn_name": "民族匹配度",
        "weight": 0.02,
        "baseline": 0.50,
        "description": "候选人民族与查询要求的匹配程度",
    },
}

# 动态特征的默认配置（未在 DYNAMIC_FEATURE_REGISTRY 中注册的属性使用此默认值）
DYNAMIC_FEATURE_DEFAULTS = {
    "weight": 0.05,
    "baseline": 0.50,
}

# 基础特征（始终在 SHAP 图中展示的特征）
SHAP_BASE_FEATURES = ["education_match", "cultural_fit", "growth_potential"]

# 动态特征 → 触发关键词（query 中出现这些关键词时，对应特征才纳入 SHAP 图）
SHAP_DYNAMIC_FEATURE_TRIGGERS = {
    "skill_match": ["技能", "技术", "会", "熟悉", "精通", "掌握", "Java", "Python", "Go",
                    "React", "Vue", "Spring", "MySQL", "Redis", "Kafka", "Docker",
                    "K8s", "PyTorch", "TensorFlow", "前端", "后端", "算法", "AI", "开发",
                    "测试", "运维", "数据", "全栈", "架构"],
    "experience_match": ["经验", "年", "工作", "资深", "高级", "senior", "初级", "junior",
                         "工龄", "年限"],
    "salary_match": ["薪资", "薪酬", "工资", "待遇", "年薪", "月薪", "k", "K", "万"],
    "location_match": ["北京", "上海", "深圳", "杭州", "广州", "成都", "南京", "武汉",
                       "西安", "苏州", "地点", "城市", "地域", "坐标"],
    "industry_match": ["行业", "领域", "互联网", "金融", "电商", "教育", "医疗", "游戏",
                       "物流", "制造", "零售"],
    "project_relevance": ["项目", "经历", "做过", "负责", "参与", "主导"],
    "certification_match": ["证书", "认证", "PMP", "CPA", "资格", "执照"],
    "language_match": ["英语", "日语", "外语", "语言", "雅思", "托福", "英文"],
    "management_match": ["管理", "带团队", "团队", "负责人", "leader", "主管", "总监"],
}

# 保持向后兼容（旧代码使用 SHAP_FEATURE_NAMES 的地方）
SHAP_FEATURE_NAMES = SHAP_FEATURE_KEYS

# ── CatBoost 模型配置 ─────────────────────────────────────────────────────────
CATBOOST_ITERATIONS = 500
CATBOOST_LEARNING_RATE = 0.05
CATBOOST_DEPTH = 6

# ── 服务配置 ──────────────────────────────────────────────────────────────────
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_HOST = "0.0.0.0"
FRONTEND_PORT = 9030
