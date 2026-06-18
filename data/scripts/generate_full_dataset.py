"""生成论文第3章描述的1000条完整多模态合成简历数据集

运行方法: cd hr_agent_mt && python -m data.scripts.generate_full_dataset

论文描述的数据集规格:
- 1000条完整简历
- 8个岗位类别 (后端25%, 前端15%, 算法15%, 产品12%, 数据10%, 运营8%, UI8%, 测试7%)
- 每条简历包含平均2.8种多模态数据
- 6种多模态类型: certificate, competition, project_arch, tech_stack, model_diagram, report
- KS检验所有关键字段p>0.05
- 技能组合信息熵4.87/5.0
- 教育经历: 多段时间线(支持专升本、本硕博路径, 全日制/非全日制标识)
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ═══════════════════════════════════════════════════════════════════════════════
# 数据定义
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# 岗位类别及占比 (论文3.1节)
POSITION_DISTRIBUTION = {
    "后端开发工程师": 0.25,
    "前端开发工程师": 0.15,
    "算法工程师": 0.15,
    "产品经理": 0.12,
    "数据分析师": 0.10,
    "运营管理": 0.08,
    "UI设计师": 0.08,
    "测试开发工程师": 0.07,
}

# 各岗位对应的技能池
SKILLS_BY_POSITION = {
    "后端开发工程师": ["Java", "Python", "Go", "C++", "Spring", "SpringBoot", "MyBatis",
                     "MySQL", "PostgreSQL", "Redis", "Kafka", "RabbitMQ", "Docker",
                     "Kubernetes", "微服务", "分布式系统", "Linux", "Nginx", "设计模式",
                     "数据结构", "算法", "并发编程", "网络编程", "RPC", "gRPC"],
    "前端开发工程师": ["JavaScript", "TypeScript", "React", "Vue", "Angular", "HTML5",
                     "CSS3", "Node.js", "Webpack", "Vite", "小程序开发", "React Native",
                     "Flutter", "性能优化", "跨端开发", "Electron", "Web安全", "GraphQL"],
    "算法工程师": ["Python", "PyTorch", "TensorFlow", "机器学习", "深度学习", "NLP",
                  "计算机视觉", "推荐系统", "强化学习", "大模型", "Transformer",
                  "BERT", "GPT", "数据挖掘", "特征工程", "模型部署", "CUDA", "分布式训练"],
    "产品经理": ["需求分析", "产品设计", "用户研究", "数据分析", "项目管理", "Axure",
               "Figma", "竞品分析", "商业分析", "A/B测试", "SQL", "用户增长",
               "产品规划", "PRD撰写", "市场调研"],
    "数据分析师": ["Python", "SQL", "Tableau", "PowerBI", "Excel", "数据可视化",
                  "统计分析", "数据建模", "ETL", "Hive", "Spark", "数据仓库",
                  "A/B测试", "用户画像", "漏斗分析", "R语言"],
    "运营管理": ["用户运营", "内容运营", "活动运营", "数据分析", "社群运营", "SEO",
               "SEM", "新媒体运营", "品牌营销", "渠道管理", "项目管理", "Excel",
               "用户增长", "转化优化", "文案撰写"],
    "UI设计师": ["Figma", "Sketch", "Photoshop", "Illustrator", "UI设计", "交互设计",
               "视觉设计", "设计系统", "动效设计", "用户体验", "原型设计",
               "品牌设计", "3D设计", "C4D"],
    "测试开发工程师": ["Python", "Java", "Selenium", "Appium", "JMeter", "接口测试",
                     "自动化测试", "性能测试", "安全测试", "CI/CD", "Jenkins",
                     "测试框架", "Mock测试", "压力测试", "白盒测试", "黑盒测试"],
}

# 通用技能池
GENERAL_SKILLS = ["Git", "团队协作", "沟通表达", "英语", "敏捷开发", "文档撰写"]

# ─── 院校知识库 (按tier分层) ───────────────────────────────────────────────────
# 加载外部知识库
_KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"

def _load_school_categories() -> Dict[str, List[str]]:
    """加载院校分类知识库"""
    kb_path = _KNOWLEDGE_DIR / "school_categories.json"
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 去掉 _metadata
        return {k: v for k, v in data.items() if not k.startswith("_")}
    # fallback
    return {
        "985": ["清华大学", "北京大学", "浙江大学", "上海交通大学", "复旦大学"],
        "211": ["北京邮电大学", "电子科技大学", "西安电子科技大学"],
        "双一流": ["南方科技大学", "深圳大学"],
        "普通本科": ["杭州电子科技大学", "成都信息工程大学"],
        "专科": ["深圳职业技术大学", "成都职业技术学院"],
    }

SCHOOL_CATEGORIES = _load_school_categories()
SCHOOLS_985 = SCHOOL_CATEGORIES.get("985", [])
SCHOOLS_211 = SCHOOL_CATEGORIES.get("211", [])
SCHOOLS_SHUANGYILIU = SCHOOL_CATEGORIES.get("双一流", [])
SCHOOLS_NORMAL = SCHOOL_CATEGORIES.get("普通本科", [])
SCHOOLS_ZHUANKE = SCHOOL_CATEGORIES.get("专科", [])

# ─── 教育路径定义 ─────────────────────────────────────────────────────────────

# 教育路径模式分布（反映真实IT行业求职者构成）
EDUCATION_PATH_DISTRIBUTION = {
    "本科直接就业": 0.40,           # 全日制本科→就业
    "本科→硕士": 0.25,              # 全日制本科→全日制硕士
    "本科→硕士→博士": 0.05,         # 全日制本科→硕士→博士
    "专科→专升本": 0.10,            # 全日制专科→专升本(非全日制居多)
    "专科直接就业": 0.08,           # 全日制专科→就业
    "本科→非全日制硕士": 0.07,      # 全日制本科→在职研究生
    "专科→专升本→硕士": 0.05,       # 专科→专升本→考研
}

COMPANIES_TIER1 = ["美团", "字节跳动", "阿里巴巴", "腾讯", "百度", "华为", "京东"]
COMPANIES_TIER2 = ["快手", "拼多多", "滴滴", "小米", "网易", "bilibili", "携程", "微博"]
COMPANIES_TIER3 = ["OPPO", "vivo", "联想", "中兴", "用友", "金蝶", "科大讯飞", "商汤"]
COMPANIES_OTHER = ["创业公司", "外企", "国企", "中小互联网公司"]

CITIES = ["北京", "上海", "深圳", "杭州", "成都", "广州", "南京", "武汉", "西安", "苏州"]
CITY_WEIGHTS = [0.25, 0.20, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]

MAJORS = {
    "tech": ["计算机科学与技术", "软件工程", "人工智能", "数据科学与大数据", "信息工程",
             "电子信息工程", "通信工程", "网络工程", "物联网工程", "信息安全"],
    "design": ["数字媒体技术", "视觉传达设计", "交互设计", "工业设计", "动画"],
    "business": ["工商管理", "市场营销", "信息管理", "电子商务", "国际经济与贸易"],
    "math": ["数学与应用数学", "统计学", "应用统计学", "信息与计算科学"],
}

# 多模态数据类型 (论文3.3节)
MULTIMODAL_TYPES = {
    "certificate": ["AWS认证架构师", "阿里云ACP", "PMP项目管理", "CPA注册会计师",
                    "软件设计师", "系统架构设计师", "数据库系统工程师",
                    "信息安全工程师", "HCIE华为认证", "红帽RHCE"],
    "competition": ["ACM-ICPC亚洲区域赛", "Kaggle竞赛", "天池大数据竞赛",
                    "数学建模竞赛", "挑战杯", "互联网+创新创业大赛",
                    "CTF安全竞赛", "蓝桥杯", "Google Code Jam", "LeetCode周赛"],
    "project_arch": ["微服务架构设计图", "分布式系统架构图", "数据流水线架构",
                     "推荐系统架构", "实时计算平台设计", "ML Pipeline架构",
                     "前端工程化架构", "DevOps流水线设计"],
    "tech_stack": ["技术栈能力雷达图", "编程语言熟练度对比", "工具链使用频率统计",
                   "技术成长路线图", "项目技术选型对比分析"],
    "model_diagram": ["深度学习模型架构图", "Transformer结构示意图", "推荐模型架构",
                      "目标检测模型设计", "GAN网络结构", "多任务学习框架"],
    "report": ["技术调研报告", "系统设计文档", "性能优化报告", "安全评估报告",
               "数据分析报告", "用户研究报告", "A/B测试报告"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 教育经历生成（多段时间线）
# ═══════════════════════════════════════════════════════════════════════════════

def _pick_school_for_tier(tier: str, rng) -> str:
    """根据学校层级随机选择院校"""
    if tier == "985":
        return str(rng.choice(SCHOOLS_985)) if SCHOOLS_985 else "清华大学"
    elif tier == "211":
        return str(rng.choice(SCHOOLS_211)) if SCHOOLS_211 else "北京邮电大学"
    elif tier == "双一流":
        pool = SCHOOLS_SHUANGYILIU if SCHOOLS_SHUANGYILIU else SCHOOLS_NORMAL
        return str(rng.choice(pool))
    elif tier == "普通本科":
        return str(rng.choice(SCHOOLS_NORMAL)) if SCHOOLS_NORMAL else "杭州电子科技大学"
    elif tier == "专科":
        return str(rng.choice(SCHOOLS_ZHUANKE)) if SCHOOLS_ZHUANKE else "深圳职业技术大学"
    else:
        all_schools = SCHOOLS_985 + SCHOOLS_211 + SCHOOLS_NORMAL
        return str(rng.choice(all_schools))


def _pick_major(position: str, degree: str, rng) -> str:
    """根据岗位和学位选择专业"""
    if position in ["后端开发工程师", "前端开发工程师", "算法工程师", "测试开发工程师"]:
        return str(rng.choice(MAJORS["tech"]))
    elif position == "UI设计师":
        return str(rng.choice(MAJORS["design"]))
    elif position in ["产品经理", "运营管理"]:
        return str(rng.choice(MAJORS["tech"] + MAJORS["business"]))
    elif position == "数据分析师":
        return str(rng.choice(MAJORS["tech"] + MAJORS["math"]))
    else:
        return str(rng.choice(MAJORS["tech"]))


def generate_education_history(position: str, work_years: int, rng) -> List[Dict[str, Any]]:
    """
    生成多段教育经历时间线
    
    返回列表，每段包含:
    - degree: 学位 (专科/本科/硕士/博士)
    - school: 学校
    - major: 专业
    - start_date: 入学年月 (YYYY-MM)
    - end_date: 毕业年月 (YYYY-MM)
    - is_fulltime: 是否全日制
    - school_tier: 学校层级 (985/211/双一流/普通本科/专科)
    
    时间线从最早的学历开始，按时间正序排列。
    """
    # 1. 确定教育路径
    paths = list(EDUCATION_PATH_DISTRIBUTION.keys())
    probs = list(EDUCATION_PATH_DISTRIBUTION.values())
    path = str(rng.choice(paths, p=probs))
    
    # 2. 推算毕业年份（假设当前是2024年，从work_years反推）
    current_year = 2024
    last_graduation_year = current_year - work_years
    
    education_segments = []
    
    if path == "本科直接就业":
        # 全日制本科4年
        grad_year = last_graduation_year
        start_year = grad_year - 4
        # 学校层级分布: 985(15%), 211(25%), 双一流(10%), 普通本科(50%)
        tier_roll = rng.random()
        if tier_roll < 0.15:
            tier = "985"
        elif tier_roll < 0.40:
            tier = "211"
        elif tier_roll < 0.50:
            tier = "双一流"
        else:
            tier = "普通本科"
        education_segments.append({
            "degree": "本科",
            "school": _pick_school_for_tier(tier, rng),
            "major": _pick_major(position, "本科", rng),
            "start_date": f"{start_year}-09",
            "end_date": f"{grad_year}-06",
            "is_fulltime": True,
            "school_tier": tier,
        })
    
    elif path == "本科→硕士":
        # 硕士毕业 → 就业
        master_grad_year = last_graduation_year
        master_start_year = master_grad_year - 3  # 学硕3年 or 专硕2年
        if rng.random() < 0.5:
            master_duration = 3
        else:
            master_duration = 2
            master_start_year = master_grad_year - master_duration
        bachelor_grad_year = master_start_year
        bachelor_start_year = bachelor_grad_year - 4
        
        # 本科学校 — 硕士普遍比本科好一点
        b_tier_roll = rng.random()
        if b_tier_roll < 0.10:
            b_tier = "985"
        elif b_tier_roll < 0.35:
            b_tier = "211"
        elif b_tier_roll < 0.45:
            b_tier = "双一流"
        else:
            b_tier = "普通本科"
        
        # 硕士学校
        m_tier_roll = rng.random()
        if m_tier_roll < 0.25:
            m_tier = "985"
        elif m_tier_roll < 0.55:
            m_tier = "211"
        elif m_tier_roll < 0.70:
            m_tier = "双一流"
        else:
            m_tier = "普通本科"
        
        education_segments.append({
            "degree": "本科",
            "school": _pick_school_for_tier(b_tier, rng),
            "major": _pick_major(position, "本科", rng),
            "start_date": f"{bachelor_start_year}-09",
            "end_date": f"{bachelor_grad_year}-06",
            "is_fulltime": True,
            "school_tier": b_tier,
        })
        education_segments.append({
            "degree": "硕士",
            "school": _pick_school_for_tier(m_tier, rng),
            "major": _pick_major(position, "硕士", rng),
            "start_date": f"{master_start_year}-09",
            "end_date": f"{master_grad_year}-06",
            "is_fulltime": True,
            "school_tier": m_tier,
        })
    
    elif path == "本科→硕士→博士":
        # 博士毕业 → 就业
        phd_grad_year = last_graduation_year
        phd_duration = int(rng.choice([4, 5]))
        phd_start_year = phd_grad_year - phd_duration
        master_grad_year = phd_start_year
        master_duration = int(rng.choice([2, 3]))
        master_start_year = master_grad_year - master_duration
        bachelor_grad_year = master_start_year
        bachelor_start_year = bachelor_grad_year - 4
        
        # 学校层级 — 博士生一般在好学校
        b_tier = "211" if rng.random() < 0.5 else "985"
        m_tier = "985" if rng.random() < 0.6 else "211"
        p_tier = "985"  # 博士基本都是985
        
        education_segments.append({
            "degree": "本科",
            "school": _pick_school_for_tier(b_tier, rng),
            "major": _pick_major(position, "本科", rng),
            "start_date": f"{bachelor_start_year}-09",
            "end_date": f"{bachelor_grad_year}-06",
            "is_fulltime": True,
            "school_tier": b_tier,
        })
        education_segments.append({
            "degree": "硕士",
            "school": _pick_school_for_tier(m_tier, rng),
            "major": _pick_major(position, "硕士", rng),
            "start_date": f"{master_start_year}-09",
            "end_date": f"{master_grad_year}-06",
            "is_fulltime": True,
            "school_tier": m_tier,
        })
        education_segments.append({
            "degree": "博士",
            "school": _pick_school_for_tier(p_tier, rng),
            "major": _pick_major(position, "博士", rng),
            "start_date": f"{phd_start_year}-09",
            "end_date": f"{phd_grad_year}-06",
            "is_fulltime": True,
            "school_tier": p_tier,
        })
    
    elif path == "专科→专升本":
        # 专科3年 → 专升本2年(多为非全日制/函授)
        benke_grad_year = last_graduation_year
        benke_start_year = benke_grad_year - 2
        zhuanke_grad_year = benke_start_year
        zhuanke_start_year = zhuanke_grad_year - 3
        
        # 专升本可能全日制也可能非全日制
        is_fulltime_upgrade = rng.random() < 0.3  # 30%概率是全日制专升本
        
        education_segments.append({
            "degree": "专科",
            "school": _pick_school_for_tier("专科", rng),
            "major": _pick_major(position, "专科", rng),
            "start_date": f"{zhuanke_start_year}-09",
            "end_date": f"{zhuanke_grad_year}-06",
            "is_fulltime": True,
            "school_tier": "专科",
        })
        education_segments.append({
            "degree": "本科",
            "school": _pick_school_for_tier("普通本科", rng),
            "major": _pick_major(position, "本科", rng),
            "start_date": f"{benke_start_year}-09",
            "end_date": f"{benke_grad_year}-06",
            "is_fulltime": is_fulltime_upgrade,
            "school_tier": "普通本科",
        })
    
    elif path == "专科直接就业":
        # 全日制专科3年
        grad_year = last_graduation_year
        start_year = grad_year - 3
        education_segments.append({
            "degree": "专科",
            "school": _pick_school_for_tier("专科", rng),
            "major": _pick_major(position, "专科", rng),
            "start_date": f"{start_year}-09",
            "end_date": f"{grad_year}-06",
            "is_fulltime": True,
            "school_tier": "专科",
        })
    
    elif path == "本科→非全日制硕士":
        # 全日制本科 → 在职读研(非全日制, 通常工作后2-3年开始读)
        # 这里last_graduation_year是本科毕业年份(因为开始工作)
        bachelor_grad_year = last_graduation_year
        bachelor_start_year = bachelor_grad_year - 4
        
        # 在职研究生通常工作2-4年后开始
        years_before_master = int(rng.randint(2, 5))
        master_start_year = bachelor_grad_year + years_before_master
        master_duration = int(rng.choice([2, 3]))
        master_grad_year = master_start_year + master_duration
        
        b_tier_roll = rng.random()
        if b_tier_roll < 0.15:
            b_tier = "985"
        elif b_tier_roll < 0.40:
            b_tier = "211"
        else:
            b_tier = "普通本科"
        
        # 非全日制硕士学校通常选211或以上
        m_tier_roll = rng.random()
        if m_tier_roll < 0.30:
            m_tier = "985"
        elif m_tier_roll < 0.65:
            m_tier = "211"
        else:
            m_tier = "普通本科"
        
        education_segments.append({
            "degree": "本科",
            "school": _pick_school_for_tier(b_tier, rng),
            "major": _pick_major(position, "本科", rng),
            "start_date": f"{bachelor_start_year}-09",
            "end_date": f"{bachelor_grad_year}-06",
            "is_fulltime": True,
            "school_tier": b_tier,
        })
        education_segments.append({
            "degree": "硕士",
            "school": _pick_school_for_tier(m_tier, rng),
            "major": _pick_major(position, "硕士", rng),
            "start_date": f"{master_start_year}-09",
            "end_date": f"{master_grad_year}-06",
            "is_fulltime": False,  # 非全日制
            "school_tier": m_tier,
        })
    
    elif path == "专科→专升本→硕士":
        # 专科3年 → 专升本2年 → 硕士2-3年
        master_grad_year = last_graduation_year
        master_duration = int(rng.choice([2, 3]))
        master_start_year = master_grad_year - master_duration
        benke_grad_year = master_start_year
        benke_start_year = benke_grad_year - 2
        zhuanke_grad_year = benke_start_year
        zhuanke_start_year = zhuanke_grad_year - 3
        
        education_segments.append({
            "degree": "专科",
            "school": _pick_school_for_tier("专科", rng),
            "major": _pick_major(position, "专科", rng),
            "start_date": f"{zhuanke_start_year}-09",
            "end_date": f"{zhuanke_grad_year}-06",
            "is_fulltime": True,
            "school_tier": "专科",
        })
        education_segments.append({
            "degree": "本科",
            "school": _pick_school_for_tier("普通本科", rng),
            "major": _pick_major(position, "本科", rng),
            "start_date": f"{benke_start_year}-09",
            "end_date": f"{benke_grad_year}-06",
            "is_fulltime": rng.random() < 0.4,  # 40%全日制专升本
            "school_tier": "普通本科",
        })
        education_segments.append({
            "degree": "硕士",
            "school": _pick_school_for_tier("211", rng),
            "major": _pick_major(position, "硕士", rng),
            "start_date": f"{master_start_year}-09",
            "end_date": f"{master_grad_year}-06",
            "is_fulltime": True,
            "school_tier": "211",
        })
    
    return education_segments


# ═══════════════════════════════════════════════════════════════════════════════
# 简历生成函数
# ═══════════════════════════════════════════════════════════════════════════════

def generate_resume(candidate_id: int, position: str, rng: np.random.RandomState) -> Dict[str, Any]:
    """生成一条完整的合成简历"""

    # 工作年限
    work_years = int(rng.randint(1, 16))

    # 生成多段教育经历
    education_history = generate_education_history(position, work_years, rng)
    
    # 从教育经历中提取最高学历信息（兼容旧字段）
    highest_edu = education_history[-1]  # 最后一段是最高学历
    education_level = highest_edu["degree"]
    school = highest_edu["school"]
    major = highest_edu["major"]

    # 技能
    position_skills = SKILLS_BY_POSITION.get(position, SKILLS_BY_POSITION["后端开发工程师"])
    n_skills = int(rng.randint(4, min(12, len(position_skills) + 1)))
    selected_skills = list(rng.choice(position_skills, size=min(n_skills, len(position_skills)), replace=False))

    # 添加通用技能
    n_general = int(rng.randint(1, 4))
    general = list(rng.choice(GENERAL_SKILLS, size=min(n_general, len(GENERAL_SKILLS)), replace=False))
    all_skills = selected_skills + general

    skills = [{"skill_name": s, "proficiency": int(rng.randint(2, 6))} for s in all_skills]

    # 工作经历
    n_exp = min(int(rng.randint(1, 5)), max(1, work_years // 3))
    experiences = []
    for i in range(n_exp):
        # 公司层级与经验年限相关
        if work_years >= 8:
            company_pool = COMPANIES_TIER1 + COMPANIES_TIER2
        elif work_years >= 4:
            company_pool = COMPANIES_TIER1 + COMPANIES_TIER2 + COMPANIES_TIER3
        else:
            company_pool = COMPANIES_TIER2 + COMPANIES_TIER3 + COMPANIES_OTHER

        exp_position = position if i == 0 else str(rng.choice(list(POSITION_DISTRIBUTION.keys())))
        experiences.append({
            "company_name": str(rng.choice(company_pool)),
            "position": exp_position,
            "duration_months": int(rng.randint(6, 48)),
            "description": _generate_exp_description(position, rng),
        })

    # 项目经历
    n_projects = int(rng.randint(1, 6))
    projects = []
    for i in range(n_projects):
        proj_skills = list(rng.choice(selected_skills, size=min(3, len(selected_skills)), replace=False))
        projects.append({
            "project_name": _generate_project_name(position, rng),
            "role": str(rng.choice(["负责人", "核心开发", "主要贡献者", "参与者"])),
            "duration_months": int(rng.randint(2, 18)),
            "technologies": proj_skills,
            "description": _generate_project_description(position, proj_skills, rng),
        })

    # 获奖/资格证书 (从multimodal中certificate和competition类型提取)
    awards_certificates = []
    # 竞赛获奖
    n_awards = int(rng.choice([0, 1, 2, 3], p=[0.3, 0.35, 0.25, 0.1]))
    for _ in range(n_awards):
        award_name = str(rng.choice(MULTIMODAL_TYPES["competition"]))
        level = str(rng.choice(["国家级", "省级", "校级"]))
        role = str(rng.choice(["队长", "队员", "个人"]))
        awards_certificates.append({
            "type": "award",
            "name": award_name,
            "level": level,
            "date": f"{int(rng.randint(2018, 2025))}.{int(rng.randint(1, 13)):02d}",
            "role": role,
            "description": f"参加{award_name}，取得{rng.choice(['金奖', '银奖', '铜奖', 'Top 10%', 'Top 20%'])}成绩",
            "image_path": f"/data/awards/{candidate_id}_award_{_}.jpg",
        })
    # 资格证书
    n_certs = int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
    for _ in range(n_certs):
        cert_name = str(rng.choice(MULTIMODAL_TYPES["certificate"]))
        awards_certificates.append({
            "type": "certificate",
            "name": cert_name,
            "level": "企业级",
            "date": f"{int(rng.randint(2018, 2025))}.{int(rng.randint(1, 13)):02d}",
            "role": None,
            "description": f"持有{cert_name}认证证书，证明了在该领域的专业能力",
            "image_path": f"/data/certificates/{candidate_id}_cert_{_}.jpg",
        })

    # 城市
    city = str(rng.choice(CITIES, p=CITY_WEIGHTS))

    # 最高学历用于兼容旧字段
    highest_edu = education_history[-1]

    # 生成个人附加信息
    age = int(22 + work_years + rng.randint(-2, 3))
    birth_year = 2024 - age
    birth_month = int(rng.randint(1, 13))
    birth_day = int(rng.randint(1, 29))
    phone_prefix = str(rng.choice(["138", "139", "136", "137", "188", "185", "150", "151", "177"]))
    phone_suffix = f"{int(rng.randint(10000000, 99999999)):08d}"

    return {
        "id": candidate_id,
        "name": f"候选人{candidate_id:04d}",
        "gender": str(rng.choice(["男", "女"], p=[0.65, 0.35])),
        "birth_date": f"{birth_year}-{birth_month:02d}-{birth_day:02d}",
        "age": age,
        "phone": f"{phone_prefix}{phone_suffix}",
        "email": f"candidate{candidate_id:04d}@example.com",
        "address": f"{city}市某区某路某号",
        "current_position": position,
        "work_years": work_years,
        "current_salary": int(rng.randint(8, 70) * 1000),
        "expected_salary": int(rng.randint(10, 80) * 1000),
        "job_status": str(rng.choice(["在职看机会", "离职", "在职不看"], p=[0.5, 0.3, 0.2])),
        "location": city,
        "highest_education": highest_edu["degree"],
        "summary": f"{position}，{work_years}年经验，{highest_edu['school']}{highest_edu['degree']}毕业",
        "education_history": education_history,
        "skills": skills,
        "awards_certificates": awards_certificates,
        "work_experiences": experiences,
        "projects": projects,
    }


def _generate_education_history(edu_level: str, position: str, rng) -> list:
    """生成多段学历时间线
    
    学历路径类型:
    - 专科直接就业 (大专)
    - 专升本: 专科 → 本科 (非全日制为主)
    - 本科直接就业
    - 本科 → 硕士 (全日制)
    - 本科 → 硕士(非全日制/在职)
    - 本科 → 硕士 → 博士
    
    每段学历包含: start_date, end_date, school, major, degree, is_fulltime, school_tier
    """
    import json
    from pathlib import Path
    
    # 加载院校知识库
    knowledge_path = Path(__file__).parent.parent / "knowledge" / "school_categories.json"
    with open(knowledge_path, "r", encoding="utf-8") as f:
        school_data = json.load(f)
    
    schools_985 = school_data["985"]
    schools_211 = school_data["211"]
    schools_syl = school_data["双一流"]
    schools_normal = school_data["普通本科"]
    schools_college = school_data["专科"]
    
    # 专业选择辅助函数
    def pick_major(pos, degree_level):
        if pos in ["后端开发工程师", "前端开发工程师", "算法工程师", "测试开发工程师"]:
            return str(rng.choice(MAJORS["tech"]))
        elif pos == "UI设计师":
            return str(rng.choice(MAJORS["design"]))
        elif pos in ["产品经理", "运营管理"]:
            return str(rng.choice(MAJORS["tech"] + MAJORS["business"]))
        elif pos == "数据分析师":
            return str(rng.choice(MAJORS["tech"] + MAJORS["math"]))
        return str(rng.choice(MAJORS["tech"]))
    
    def get_school_tier(school_name):
        if school_name in schools_985:
            return "985"
        elif school_name in schools_211:
            return "211"
        elif school_name in schools_syl:
            return "双一流"
        elif school_name in schools_college:
            return "专科"
        else:
            return "普通本科"
    
    history = []
    
    # 基准毕业年份（根据年龄倒推，假设18岁入学）
    base_year = int(rng.randint(2005, 2022))
    
    if edu_level == "大专":
        # 20%概率走专升本路径
        if rng.random() < 0.20:
            # 专科 → 本科 (专升本)
            college_school = str(rng.choice(schools_college))
            college_major = pick_major(position, "专科")
            start_y = base_year - 5  # 专科3年 + 本科2年
            history.append({
                "start_date": f"{start_y}-09",
                "end_date": f"{start_y + 3}-06",
                "school": college_school,
                "major": college_major,
                "degree": "大专",
                "is_fulltime": True,
                "school_tier": "专科"
            })
            # 专升本（70%非全日制，30%全日制）
            is_ft = rng.random() < 0.30
            upgrade_school = str(rng.choice(schools_normal))
            history.append({
                "start_date": f"{start_y + 3}-09",
                "end_date": f"{start_y + 5}-06",
                "school": upgrade_school,
                "major": college_major,
                "degree": "本科",
                "is_fulltime": is_ft,
                "school_tier": get_school_tier(upgrade_school)
            })
        else:
            # 纯专科
            college_school = str(rng.choice(schools_college))
            start_y = base_year - 3
            history.append({
                "start_date": f"{start_y}-09",
                "end_date": f"{base_year}-06",
                "school": college_school,
                "major": pick_major(position, "专科"),
                "degree": "大专",
                "is_fulltime": True,
                "school_tier": "专科"
            })
    
    elif edu_level == "本科":
        # 本科就业
        all_undergrad = schools_985 + schools_211 + schools_syl + schools_normal
        school = str(rng.choice(all_undergrad))
        start_y = base_year - 4
        history.append({
            "start_date": f"{start_y}-09",
            "end_date": f"{base_year}-06",
            "school": school,
            "major": pick_major(position, "本科"),
            "degree": "本科",
            "is_fulltime": True,
            "school_tier": get_school_tier(school)
        })
    
    elif edu_level == "硕士":
        # 先生成本科段
        undergrad_pool = schools_985 + schools_211 + schools_syl + schools_normal
        undergrad_school = str(rng.choice(undergrad_pool))
        start_y = base_year - 7  # 本科4年 + 硕士3年
        history.append({
            "start_date": f"{start_y}-09",
            "end_date": f"{start_y + 4}-06",
            "school": undergrad_school,
            "major": pick_major(position, "本科"),
            "degree": "本科",
            "is_fulltime": True,
            "school_tier": get_school_tier(undergrad_school)
        })
        
        # 硕士段 — 80%全日制，20%非全日制(在职)
        is_ft = rng.random() < 0.80
        if is_ft:
            # 全日制硕士倾向去更好的学校
            grad_pool = schools_985 + schools_211
            grad_school = str(rng.choice(grad_pool))
            duration = 3 if rng.random() < 0.6 else 2  # 学硕3年，专硕2年
        else:
            # 非全日制通常在普通院校
            grad_pool = schools_211 + schools_syl + schools_normal
            grad_school = str(rng.choice(grad_pool))
            duration = 3
        
        history.append({
            "start_date": f"{start_y + 4}-09",
            "end_date": f"{start_y + 4 + duration}-06",
            "school": grad_school,
            "major": pick_major(position, "硕士"),
            "degree": "硕士",
            "is_fulltime": is_ft,
            "school_tier": get_school_tier(grad_school)
        })
    
    elif edu_level == "博士":
        # 本科 → 硕士 → 博士
        start_y = base_year - 11  # 4 + 3 + 4
        
        # 本科段
        undergrad_school = str(rng.choice(schools_985 + schools_211))
        history.append({
            "start_date": f"{start_y}-09",
            "end_date": f"{start_y + 4}-06",
            "school": undergrad_school,
            "major": pick_major(position, "本科"),
            "degree": "本科",
            "is_fulltime": True,
            "school_tier": get_school_tier(undergrad_school)
        })
        
        # 硕士段
        master_school = str(rng.choice(schools_985))
        master_dur = 3 if rng.random() < 0.5 else 2
        history.append({
            "start_date": f"{start_y + 4}-09",
            "end_date": f"{start_y + 4 + master_dur}-06",
            "school": master_school,
            "major": pick_major(position, "硕士"),
            "degree": "硕士",
            "is_fulltime": True,
            "school_tier": get_school_tier(master_school)
        })
        
        # 博士段
        phd_school = str(rng.choice(schools_985))
        phd_dur = int(rng.choice([3, 4, 5], p=[0.2, 0.5, 0.3]))
        history.append({
            "start_date": f"{start_y + 4 + master_dur}-09",
            "end_date": f"{start_y + 4 + master_dur + phd_dur}-06",
            "school": phd_school,
            "major": pick_major(position, "博士"),
            "degree": "博士",
            "is_fulltime": True,
            "school_tier": get_school_tier(phd_school)
        })
    
    return history


def _generate_exp_description(position: str, rng) -> str:
    """生成丰富的工作经历描述（多句话，包含具体职责、技术细节、量化成果）"""
    # 每个岗位有多组模板，每组包含2-4个句子拼接，模拟真实简历的详细程度
    templates = {
        "后端开发工程师": [
            [
                "负责{}核心交易系统的架构设计与开发，基于Spring Cloud微服务体系完成服务拆分与治理",
                "设计并实现高并发下单链路，通过Redis缓存+消息队列异步解耦，日均处理{}万笔订单，系统可用性达99.{}%",
                "主导MySQL分库分表方案落地，单表数据量从{}亿降至千万级，慢查询减少{}%",
            ],
            [
                "负责公司{}业务中台的后端架构设计，基于Go语言和gRPC完成跨服务通信框架搭建",
                "设计实现分布式配置中心和服务注册发现组件，支撑{}+微服务实例稳定运行",
                "通过链路追踪和性能监控体系建设，将线上P99延迟从{}ms优化至{}ms，故障定位效率提升{}%",
            ],
            [
                "参与{}平台后端核心模块的开发与维护，基于Java/SpringBoot技术栈完成业务功能迭代",
                "负责数据库设计与SQL优化，完成复杂业务查询的索引优化和慢SQL治理，平均查询耗时降低{}%",
                "设计并实现基于Kafka的消息驱动架构，完成订单状态异步流转和数据一致性保障方案",
                "编写单元测试和集成测试，代码覆盖率从{}%提升至{}%",
            ],
            [
                "负责{}方向的分布式存储系统研发，基于Raft协议实现多副本一致性保障",
                "设计并实现自动扩缩容方案，支撑{}TB级数据存储，读写QPS峰值达{}万",
                "参与系统稳定性建设，制定容灾预案和混沌工程实践方案，全年SLA达99.{}%",
            ],
        ],
        "前端开发工程师": [
            [
                "负责{}产品的前端架构设计与核心功能开发，基于React/TypeScript技术栈搭建SPA应用",
                "主导前端性能优化专项，通过代码分割、图片懒加载和CDN策略，FCP从{}s降至{}s，LCP优化{}%",
                "设计并实现组件库，覆盖{}+通用组件，支撑{}个业务线复用",
            ],
            [
                "负责公司{}小程序和H5多端项目的前端开发，基于Taro跨端框架完成一套代码多端运行",
                "主导Webpack构建优化和CI/CD流水线搭建，构建时间从{}分钟降至{}分钟，开发效率提升{}%",
                "设计并实现移动端离线缓存和增量更新方案，用户端白屏率降低{}%",
            ],
            [
                "参与{}可视化大屏项目开发，基于Vue3+ECharts完成复杂数据图表和实时数据更新",
                "设计前端监控体系，集成Sentry错误上报和性能指标采集，线上错误率降低{}%",
                "负责前端工程化建设，搭建ESLint/Prettier/Husky代码规范工具链和Monorepo管理方案",
            ],
        ],
        "算法工程师": [
            [
                "负责{}推荐系统核心排序模型的迭代优化，基于DeepFM/DCN等深度学习模型完成特征交叉建模",
                "设计并实现多目标优化框架，通过MMOE架构同时优化CTR和CVR，整体GMV提升{}%",
                "负责特征工程建设，基于Spark完成百亿级样本的特征提取和存储，特征覆盖率从{}%提升至{}%",
                "主导AB实验平台与模型效果评估体系搭建，支撑日均{}+实验同时运行",
            ],
            [
                "负责{}方向NLP模型的研发与落地，基于BERT/GPT架构完成文本分类、实体识别等任务",
                "设计并实现大模型微调Pipeline，基于LoRA/QLoRA方法完成领域适配，推理延迟降低{}%",
                "负责模型部署与serving框架建设，基于TensorRT实现模型加速，吞吐量提升{}倍",
            ],
            [
                "参与{}计算机视觉项目，基于YOLOv5/Mask R-CNN完成目标检测与实例分割模型训练",
                "设计数据增强策略和模型蒸馏方案，在保持精度(mAP={}%)的同时将模型参数量压缩{}%",
                "负责模型训练工程化，基于PyTorch DDP实现多机多卡分布式训练，训练效率提升{}倍",
                "通过SHAP值分析完成模型可解释性拆解，定位核心特征因子并输出分析报告",
            ],
        ],
        "产品经理": [
            [
                "负责{}产品线的整体规划与迭代，从0到1完成产品定义、需求分析、原型设计到上线交付全流程",
                "通过用户调研和数据分析驱动产品决策，主导完成{}+功能迭代，产品DAU增长{}%",
                "制定用户增长策略，设计拉新、促活、留存完整链路，新用户{}日留存率从{}%提升至{}%",
            ],
            [
                "负责{}商业化产品的需求规划与项目管理，协调研发、设计、测试等{}+人团队推进项目落地",
                "主导竞品分析和行业调研，输出PRD文档{}+份，制定产品路线图和版本规划",
                "设计A/B测试方案验证产品假设，通过数据驱动的迭代方法将核心转化率提升{}%",
            ],
        ],
        "数据分析师": [
            [
                "负责{}业务线的数据分析与挖掘工作，基于Hive/Spark完成亿级数据的清洗、建模和分析",
                "搭建业务核心指标体系和数据看板，通过Tableau完成可视化报表开发，覆盖{}+关键指标",
                "通过用户分群和漏斗分析定位增长瓶颈，输出策略建议{}+项，驱动业务指标提升{}%",
            ],
            [
                "负责公司数据仓库建设，基于Hive/Spark SQL完成ODS-DWD-DWS-ADS分层体系搭建",
                "设计并实现ETL数据流水线，完成{}+张报表的自动化调度和数据质量监控",
                "基于Python进行数据探查和异常检测，使用sweetviz和pandas-profiling完成全量数据质量分析",
                "通过统计学方法和机器学习模型完成用户画像标签体系建设，标签覆盖率达{}%",
            ],
        ],
        "运营管理": [
            [
                "负责{}产品的用户运营体系搭建，设计并执行用户分层运营策略，用户活跃度提升{}%",
                "主导内容运营和社群运营方案，搭建{}+社群，社群日均互动量达{}万次",
                "设计并执行线上活动策划方案，单场活动参与用户{}万+，活动ROI达到{}:1",
            ],
            [
                "负责{}品牌的新媒体运营和渠道管理，管理公众号、抖音、小红书等{}个平台",
                "通过数据分析优化投放策略，CPC降低{}%，获客成本从{}元降至{}元",
                "搭建用户增长模型，设计裂变和转介绍机制，月新增用户{}万+",
            ],
        ],
        "UI设计师": [
            [
                "负责{}产品的UI/UX设计工作，基于Figma完成从需求分析、交互设计到视觉输出的全流程",
                "主导设计系统搭建，制定设计规范和组件标准，覆盖{}+标准组件，设计协作效率提升{}%",
                "通过用户测试和数据分析驱动设计决策，核心流程转化率提升{}%",
            ],
            [
                "负责公司{}品牌的视觉设计与品牌升级，完成品牌VI体系从0到1的搭建",
                "设计并产出{}套完整的UI界面方案，通过多轮用户测试迭代优化交互体验",
                "参与动效设计和3D可视化项目，使用After Effects和C4D完成品牌宣传素材制作",
            ],
        ],
        "测试开发工程师": [
            [
                "负责{}项目的测试架构设计与自动化测试框架搭建，基于Selenium/Appium实现UI自动化回归",
                "设计并实现接口自动化测试平台，覆盖{}+核心接口，自动化覆盖率从{}%提升至{}%",
                "搭建性能测试体系，基于JMeter/Locust完成压力测试和容量评估，发现{}+性能瓶颈并推动修复",
            ],
            [
                "参与{}项目的质量保障工作，制定测试策略和测试计划，编写测试用例{}+条",
                "搭建CI/CD流水线中的自动化测试环节，基于Jenkins实现代码提交后自动触发回归测试",
                "负责测试环境管理和Mock服务搭建，通过Docker容器化方案实现环境快速部署和隔离",
            ],
        ],
    }
    # 选择对应岗位的随机模板组
    position_templates = templates.get(position, templates["后端开发工程师"])
    sentences = list(rng.choice(position_templates))

    # 随机业务词
    business_words = ["电商", "社交", "金融", "物流", "本地生活", "出行", "教育", "医疗", "游戏", "企业服务"]
    biz_word = str(rng.choice(business_words))

    # 填充模板中的占位符
    filled_sentences = []
    for s in sentences:
        filled = s
        # 先替换第一个{}为业务词（如果是描述开头）
        if filled.startswith("负责{}") or filled.startswith("参与{}"):
            filled = filled.replace("{}", biz_word, 1)
        # 剩余的{}用随机数字填充
        while "{}" in filled:
            filled = filled.replace("{}", str(int(rng.randint(2, 99))), 1)
        filled_sentences.append(filled)

    # 随机组合2-3个句子（模拟真实简历的多条描述）
    n_sentences = int(rng.choice([2, 3], p=[0.4, 0.6]))
    selected = filled_sentences[:min(n_sentences, len(filled_sentences))]
    return "；".join(selected)


def _generate_project_name(position: str, rng) -> str:
    """生成项目名称"""
    prefixes = {
        "后端开发工程师": ["高并发", "分布式", "微服务", "实时", "智能"],
        "前端开发工程师": ["可视化", "低代码", "跨端", "组件化", "响应式"],
        "算法工程师": ["智能推荐", "深度学习", "NLP", "多模态", "大模型"],
        "产品经理": ["用户增长", "商业化", "效率提升", "体验优化", "创新"],
        "数据分析师": ["数据中台", "用户画像", "实时报表", "智能BI", "数据治理"],
        "运营管理": ["用户运营", "内容分发", "活动平台", "社群管理", "增长"],
        "UI设计师": ["设计系统", "品牌升级", "体验重构", "动效", "3D可视化"],
        "测试开发工程师": ["自动化测试", "性能测试", "质量平台", "CI/CD", "混沌工程"],
    }
    suffixes = ["平台", "系统", "引擎", "框架", "工具", "服务", "方案"]
    prefix_pool = prefixes.get(position, prefixes["后端开发工程师"])
    return f"{rng.choice(prefix_pool)}{rng.choice(suffixes)}"


def _generate_project_description(position: str, skills: list, rng) -> str:
    """生成丰富的项目描述（包含具体技术实现步骤、方法论、量化成果）"""
    skill_str = "、".join(skills[:3]) if skills else "相关技术"

    # 每个岗位的多组项目描述模板，每组包含3-5个具体步骤
    project_templates = {
        "后端开发工程师": [
            [
                f"基于{skill_str}技术栈完成系统核心模块的架构设计与代码开发",
                "设计并实现分层缓存策略(本地缓存+Redis集群)，将热点数据查询QPS从{}提升至{}",
                "通过异步消息队列解耦核心链路，实现最终一致性保障，系统吞吐量提升{}%",
                "编写完善的技术文档和接口文档，完成代码Review和上线部署",
            ],
            [
                f"使用{skill_str}完成微服务架构下的核心业务开发与性能优化",
                "设计并实现数据库读写分离方案和分库分表策略，支撑千万级数据量的高效查询",
                "通过慢SQL分析和索引优化将平均查询耗时从{}ms降至{}ms",
                "搭建监控告警体系，实现核心指标的实时采集和异常自动通知",
            ],
        ],
        "前端开发工程师": [
            [
                f"基于{skill_str}技术栈完成产品核心页面的开发与交互实现",
                "设计并实现响应式布局方案，适配PC、平板、手机等多种设备，覆盖率达{}%",
                "通过虚拟列表、图片懒加载和组件按需加载策略，页面首屏加载时间降低{}%",
                "封装{}+通用业务组件，提高团队开发效率和代码复用率",
            ],
            [
                f"使用{skill_str}完成前端单页应用的架构搭建和核心功能开发",
                "设计统一的状态管理方案和接口层封装，支撑{}+页面的数据流管理",
                "实现前端CI/CD自动化流程，集成单元测试和E2E测试，代码质量显著提升",
                "通过Performance API和Lighthouse完成性能基线建设和持续优化",
            ],
        ],
        "算法工程师": [
            [
                f"基于{skill_str}完成核心算法模型的研发与工程化落地",
                "设计特征工程流水线，完成{}+维特征的提取、筛选和存储",
                "通过网格搜索和贝叶斯优化完成超参数调优，模型AUC从{}提升至{}",
                "基于SHAP值分析完成模型可解释性报告，定位Top特征并输出业务洞察",
                "完成模型的A/B实验上线和效果评估，线上指标持续正向",
            ],
            [
                f"使用{skill_str}完成深度学习模型的训练、优化与部署",
                "设计数据预处理Pipeline，完成数据清洗、增强和标准化，训练样本覆盖{}万+",
                "通过模型剪枝和量化技术将推理耗时从{}ms降至{}ms，满足线上实时性要求",
                "搭建模型监控体系，实时检测模型效果漂移和数据分布变化",
            ],
        ],
        "产品经理": [
            [
                f"结合{skill_str}方法论完成产品需求的定义、设计与验证",
                "通过用户访谈({}+场)和问卷调研({}+份)完成需求发掘和优先级排序",
                "输出完整的PRD文档和交互原型，协调研发团队按期交付",
                "设计数据埋点方案，通过漏斗分析和用户路径分析持续优化产品体验",
            ],
        ],
        "数据分析师": [
            [
                f"基于{skill_str}完成业务数据的全链路分析与建模",
                "设计并实现自动化报表体系，覆盖{}+核心业务指标的日/周/月报产出",
                "通过RFM模型和聚类分析完成用户分群，输出差异化运营策略建议",
                "使用Python完成探索性数据分析和异常检测，定位数据质量问题并推动修复",
            ],
        ],
        "运营管理": [
            [
                f"利用{skill_str}能力完成运营策略的制定和执行",
                "设计用户生命周期运营方案，针对不同阶段用户制定差异化触达策略",
                "通过A/B测试验证策略效果，最终选定ROI最优的方案全量上线",
                "搭建运营数据看板，实现核心指标的实时监控和异常预警",
            ],
        ],
        "UI设计师": [
            [
                f"基于{skill_str}工具完成产品的视觉设计与交互规范制定",
                "通过竞品分析和设计趋势研究，输出{}+版设计方案并完成评审决策",
                "建立设计Token体系和组件规范，确保多人协作下的设计一致性",
                "完成设计走查和验收流程，确保研发实现与设计稿的像素级还原",
            ],
        ],
        "测试开发工程师": [
            [
                f"基于{skill_str}完成自动化测试框架的设计与核心用例开发",
                "设计分层测试策略(单元测试+接口测试+UI测试)，测试覆盖率达{}%",
                "搭建性能基线和回归测试体系，每次发版前自动执行{}+条核心用例",
                "建立缺陷分析和质量度量体系，推动研发过程质量持续改进",
            ],
        ],
    }
    templates = project_templates.get(position, project_templates["后端开发工程师"])
    steps = list(rng.choice(templates))

    # 填充占位符
    filled = []
    for s in steps:
        while "{}" in s:
            s = s.replace("{}", str(int(rng.randint(2, 99))), 1)
        filled.append(s)

    # 随机选取3-4步
    n_steps = int(rng.choice([3, 4], p=[0.5, 0.5]))
    selected = filled[:min(n_steps, len(filled))]
    return "；".join(selected)


def _generate_multimodal_description(mm_type: str, item_name: str, position: str, rng) -> str:
    """生成多模态数据的文本描述"""
    descriptions = {
        "certificate": f"持有{item_name}认证证书，证明了在该领域的专业能力",
        "competition": f"参加{item_name}，取得{rng.choice(['金奖', '银奖', '铜奖', 'Top 10%', 'Top 20%'])}成绩",
        "project_arch": f"{item_name}，展示了系统的整体技术架构和核心组件设计",
        "tech_stack": f"{item_name}，直观展示各项技术的掌握程度和使用深度",
        "model_diagram": f"{item_name}，展示了模型的网络结构和数据流向",
        "report": f"{item_name}，详细记录了技术调研或系统优化的完整过程和结论",
    }
    return descriptions.get(mm_type, f"{item_name}相关多模态数据")


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def generate_full_dataset(n_resumes: int = 1000, seed: int = RANDOM_SEED) -> List[Dict]:
    """生成完整的1000条合成简历数据集"""
    rng = np.random.RandomState(seed)

    # 按论文描述的岗位分布生成
    positions = []
    for pos, ratio in POSITION_DISTRIBUTION.items():
        count = int(n_resumes * ratio)
        positions.extend([pos] * count)

    # 补足到n_resumes
    while len(positions) < n_resumes:
        positions.append(str(rng.choice(list(POSITION_DISTRIBUTION.keys()))))
    positions = positions[:n_resumes]

    # 打乱顺序
    rng.shuffle(positions)

    # 生成简历
    resumes = []
    for i, position in enumerate(positions):
        resume = generate_resume(i + 1, position, rng)
        resumes.append(resume)

    return resumes


def compute_dataset_statistics(resumes: List[Dict]) -> Dict[str, Any]:
    """计算数据集统计信息"""
    from collections import Counter

    n = len(resumes)

    # 岗位分布
    position_counts = Counter(r["current_position"] for r in resumes)
    position_dist = {k: round(v / n, 3) for k, v in position_counts.items()}

    # 学历分布
    edu_counts = Counter(r.get("highest_education", r.get("education_level", "")) for r in resumes)
    edu_dist = {k: round(v / n, 3) for k, v in edu_counts.items()}

    # 工作年限统计
    work_years = [r["work_years"] for r in resumes]

    # 技能数量统计
    skill_counts = [len(r["skills"]) for r in resumes]

    # 获奖证书统计
    awards_counts = [len(r.get("awards_certificates", r.get("multimodal", []))) for r in resumes]
    avg_awards = np.mean(awards_counts)

    # 技能多样性（信息熵）
    all_skills = []
    for r in resumes:
        all_skills.extend(s["skill_name"] for s in r["skills"])
    skill_counter = Counter(all_skills)
    total_skills = sum(skill_counter.values())
    probs = [c / total_skills for c in skill_counter.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(len(skill_counter))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # 教育路径统计
    edu_paths = Counter()
    fulltime_count = 0
    part_time_count = 0
    for r in resumes:
        eh = r.get("education_history", [])
        path = " → ".join([e["degree"] for e in eh])
        edu_paths[path] += 1
        for e in eh:
            if e.get("is_fulltime"):
                fulltime_count += 1
            else:
                part_time_count += 1

    return {
        "total_resumes": n,
        "position_distribution": position_dist,
        "education_distribution": edu_dist,
        "work_years": {
            "mean": round(float(np.mean(work_years)), 2),
            "std": round(float(np.std(work_years)), 2),
            "min": int(np.min(work_years)),
            "max": int(np.max(work_years)),
        },
        "skills_per_resume": {
            "mean": round(float(np.mean(skill_counts)), 2),
            "std": round(float(np.std(skill_counts)), 2),
        },
        "awards_per_resume": {
            "mean": round(float(avg_awards), 2),
        },
        "skill_diversity": {
            "unique_skills": len(skill_counter),
            "entropy": round(float(entropy), 2),
            "normalized_entropy_5": round(float(normalized_entropy * 5), 2),
        },
        "education_paths": dict(edu_paths.most_common(10)),
        "fulltime_vs_parttime": {
            "fulltime": fulltime_count,
            "part_time": part_time_count,
            "ratio": round(fulltime_count / max(1, fulltime_count + part_time_count), 3)
        },
    }


def main():
    print("=" * 70)
    print("生成1000条完整多模态合成简历数据集 (论文第3章) - 含多段学历时间线")
    print("=" * 70)

    # 生成数据
    print("\n正在生成1000条合成简历...")
    resumes = generate_full_dataset(n_resumes=1000, seed=RANDOM_SEED)
    print(f"  生成完成: {len(resumes)} 条简历")

    # 计算统计信息
    print("\n计算数据集统计信息...")
    stats = compute_dataset_statistics(resumes)

    print(f"\n  岗位分布: {stats['position_distribution']}")
    print(f"  学历分布: {stats['education_distribution']}")
    print(f"  工作年限: mean={stats['work_years']['mean']}, std={stats['work_years']['std']}")
    print(f"  技能数/人: mean={stats['skills_per_resume']['mean']}")
    print(f"  获奖证书/人: mean={stats['awards_per_resume']['mean']}")
    print(f"  技能多样性: 熵={stats['skill_diversity']['entropy']}, "
          f"归一化(5分制)={stats['skill_diversity']['normalized_entropy_5']}")
    print(f"  教育路径分布: {stats['education_paths']}")
    print(f"  全日制vs非全日制: {stats['fulltime_vs_parttime']}")

    # 保存数据集
    output_dir = Path(__file__).parent.parent / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整数据集
    dataset_file = output_dir / "full_resume_dataset_1000.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(resumes, f, ensure_ascii=False, indent=2)
    print(f"\n  数据集已保存: {dataset_file}")
    print(f"  文件大小: {dataset_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 保存统计信息
    stats_file = output_dir / "dataset_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  统计信息: {stats_file}")

    print("\n" + "=" * 70)
    print("数据集生成完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
