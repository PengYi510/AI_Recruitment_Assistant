"""合成简历生成Skill - 文本优先流水线

新流程（区别于旧版直接生成 JSON 入库）：
1. 随机生成一个候选人"画像骨架"（学历/学校档次/方向/公司/年限等），保证多样性与可控性
2. 调用 LLM（LongCat）基于骨架写出一大段「自然语言简历文本」（4000+ 字，风格对标真实简历，
   尤其项目经历部分要有 5-7 条详细技术描述）
3. 将简历原文交给 ResumeExtractionSkill，从文本中提取教育/工作/技能/项目等结构化字段并入库 + 向量化
4. 同时把简历原文落盘到 data/synthetic/resumes/，便于复查

LLM 失败时降级为本地模板长文本，保证批量任务不中断。
"""
import logging
import random
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

from backend.skills.base_skill import BaseSkill
from backend.config import SYNTHETIC_DIR

logger = logging.getLogger(__name__)


# ── 海外院校库（从 overseas_school_rankings.json 加载，按等价层级分组）──────────
_OVERSEAS_KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "overseas_school_rankings.json"


def _load_overseas_pools() -> Dict[str, List[str]]:
    """加载海外院校并按等价层级（985/211/双一流）分组，校名取 '英文(中文)' display 形式。

    注意：港/澳/台 院校（country 含「中国」）虽在 QS/US News 排名内，但属中国境内，
    不作为「留学经历」的海外目的地，这里从生成池中剔除（其等价层级仍由 school_tier 保留）。
    """
    pools: Dict[str, List[str]] = {"985": [], "211": [], "双一流": []}
    seen = set()
    try:
        data = json.loads(_OVERSEAS_KB_PATH.read_text(encoding="utf-8"))
        for uni in data.get("universities", []):
            tier = uni.get("equiv_tier")
            country = uni.get("country", "")
            if "中国" in country:  # 剔除港澳台，避免被写成「出国留学」
                continue
            disp = uni.get("display") or f"{uni.get('name_en','')}({uni.get('name_cn','')})"
            if tier in pools and disp not in seen:
                seen.add(disp)
                pools[tier].append(disp)
    except Exception as e:  # pragma: no cover
        logger.error(f"[Generator] 海外院校库加载失败: {e}")
    # 兜底：避免空池
    for k, v in pools.items():
        if not v:
            v.append("University of Example(示例大学)")
    return pools


OVERSEAS_POOLS = _load_overseas_pools()
# 仅含英国/美国/澳洲/港新加等英语区，用于挑选"水硕"高频目的地（QS51-150 多为此类）
OVERSEAS_ALL = OVERSEAS_POOLS["985"] + OVERSEAS_POOLS["211"] + OVERSEAS_POOLS["双一流"]

# ── 画像骨架候选项（仅用于驱动 LLM 写作，保证多样性）────────────────────────
EDUCATION_LEVELS = ["博士", "硕士", "本科", "大专"]
# 按权威层级分组（与 school_categories.json 保持一致，驱动多样性）
SCHOOLS_985 = ["清华大学", "北京大学", "浙江大学", "上海交通大学", "复旦大学",
               "南京大学", "中国科学技术大学", "武汉大学", "华中科技大学", "西安交通大学",
               "中国科学院大学", "哈尔滨工业大学", "中山大学", "四川大学", "同济大学"]
SCHOOLS_211 = ["北京邮电大学", "西安电子科技大学", "南京航空航天大学", "华南理工大学",
               "东南大学", "重庆大学", "湖南大学", "大连理工大学", "北京交通大学"]
SCHOOLS_SHUANG = ["深圳大学", "杭州电子科技大学", "南京邮电大学", "广州大学"]
SCHOOLS_NORMAL = ["成都信息工程大学", "广东工业大学", "山西大学", "江苏大学",
                  "河南工业大学", "湖北工业大学", "重庆邮电大学", "沈阳工业大学"]
SCHOOLS_ZHUANKE = ["深圳职业技术大学", "金华职业技术学院", "南京信息职业技术学院",
                   "广东轻工职业技术学院", "天津电子信息职业技术学院"]
MAJORS = ["计算机科学与技术", "软件工程", "人工智能", "数据科学与大数据技术",
          "电子信息工程", "计算机技术", "统计学", "信息与通信工程"]
DIRECTIONS = ["后端开发", "大数据开发", "机器学习算法", "数据挖掘", "推荐算法",
              "搜索算法", "NLP/大模型应用", "全栈开发", "数据分析", "AI工程"]
TECH_SKILLS = ["Java", "Python", "Go", "C++", "JavaScript", "TypeScript", "React", "Vue",
               "Spring Boot", "Django", "FastAPI", "Flask", "Docker", "Kubernetes",
               "MySQL", "Redis", "MongoDB", "HBase", "Oracle", "Kafka", "Flink", "Spark",
               "Hadoop", "Hive", "Elasticsearch", "TensorFlow", "PyTorch", "LangChain",
               "LangGraph", "Scala", "Linux", "Git"]
COMPANIES = ["美团", "字节跳动", "阿里巴巴", "腾讯", "京东", "百度", "华为", "小米",
             "网易", "快手", "滴滴", "拼多多", "携程", "蚂蚁集团", "辰安科技", "中软融鑫"]
CITIES = ["北京", "上海", "深圳", "杭州", "成都", "广州", "南京", "武汉", "西安"]
# 现居地/家乡池（省份/城市格式）
CITY_LOCATIONS = [
    "北京/北京", "上海/上海", "广东/深圳", "广东/广州", "浙江/杭州",
    "四川/成都", "江苏/南京", "湖北/武汉", "陕西/西安", "重庆/重庆",
    "天津/天津", "湖南/长沙", "福建/厦门", "山东/青岛", "辽宁/大连",
    "江苏/苏州", "浙江/宁波", "安徽/合肥", "河南/郑州", "广东/东莞",
]
HOMETOWN_LOCATIONS = [
    "湖南/长沙", "四川/成都", "河南/郑州", "湖北/武汉", "安徽/合肥",
    "江西/南昌", "山东/济南", "河北/石家庄", "广东/广州", "浙江/杭州",
    "江苏/南京", "福建/福州", "陕西/西安", "辽宁/沈阳", "黑龙江/哈尔滨",
    "吉林/长春", "山西/太原", "贵州/贵阳", "云南/昆明", "广西/南宁",
    "甘肃/兰州", "内蒙古/呼和浩特", "新疆/乌鲁木齐", "重庆/重庆",
]
# 海外城市（用于留学生的现居地）
OVERSEAS_CITY_LOCATIONS = [
    "美国/旧金山", "美国/纽约", "美国/洛杉矶", "美国/西雅图", "美国/波士顿",
    "英国/伦敦", "英国/曼彻斯特", "新加坡/新加坡", "日本/东京", "澳大利亚/悉尼",
    "加拿大/多伦多", "加拿大/温哥华", "德国/柏林", "法国/巴黎",
]
POSITIONS = ["高级工程师", "技术专家", "架构师", "开发工程师", "技术总监",
             "算法工程师", "数据挖掘工程师", "数据分析师", "ETL工程师", "实习生"]
AWARD_POOL = ["全国大学生数学建模竞赛国家二等奖", "全国大学生市场调研竞赛国家三等奖",
              "ACM-ICPC区域赛金奖", "蓝桥杯省一等奖", "软件设计师", "系统架构师",
              "PMP认证", "AWS解决方案架构师认证", "CET-6", "IELTS 7.0", "发明专利", "优秀毕业生"]
JOB_STATUS = ["在职看机会", "离职", "在职不看", "应届", "实习"]

# ── 学术论文与国际会议知识库（从 academic_venues.json 加载）────────────────────
_ACADEMIC_KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "academic_venues.json"


def _load_academic_venues() -> Dict[str, Any]:
    """加载学术场所知识库"""
    try:
        data = json.loads(_ACADEMIC_KB_PATH.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        logger.error(f"[Generator] 学术场所知识库加载失败: {e}")
        return {"journals": [], "conferences": []}


_ACADEMIC_VENUES = _load_academic_venues()

# 按等级分组的期刊和会议
JOURNALS_CCF_A = [j for j in _ACADEMIC_VENUES.get("journals", []) if j.get("rank") == "CCF-A"]
JOURNALS_CCF_B = [j for j in _ACADEMIC_VENUES.get("journals", []) if j.get("rank") == "CCF-B"]
JOURNALS_CCF_C = [j for j in _ACADEMIC_VENUES.get("journals", []) if j.get("rank") == "CCF-C"]
CONFERENCES_CCF_A = [c for c in _ACADEMIC_VENUES.get("conferences", []) if c.get("rank") == "CCF-A"]
CONFERENCES_CCF_B = [c for c in _ACADEMIC_VENUES.get("conferences", []) if c.get("rank") == "CCF-B"]
CONFERENCES_CCF_C = [c for c in _ACADEMIC_VENUES.get("conferences", []) if c.get("rank") == "CCF-C"]

# 论文标题模板（用于生成逼真的论文标题）
PAPER_TITLE_TEMPLATES = [
"A {adj} Approach to {topic} via {method}",
"{method}-based {topic} for {app}",
"Efficient {topic} with {method} in {app}",
"Learning {topic} Representations using {method}",
"{method} for Large-Scale {topic}: A {adj} Framework",
"Towards {adj} {topic}: {method} with {app}",
"Rethinking {topic} through {method}",
"{adj} {method} for {topic} in {app}",
]
PAPER_TOPICS = ["Recommendation", "Graph Neural Networks", "Knowledge Graphs",
                "Text Classification", "Anomaly Detection", "Feature Selection",
                "Time Series Forecasting", "Multi-Modal Learning", "Federated Learning",
                "Reinforcement Learning", "Semantic Segmentation", "Object Detection",
                "Named Entity Recognition", "Sentiment Analysis", "Data Augmentation"]
PAPER_METHODS = ["Transformer", "Attention Mechanism", "Contrastive Learning",
                 "Graph Attention Network", "Variational Autoencoder", "Diffusion Model",
                 "Meta-Learning", "Prompt Tuning", "Knowledge Distillation",
                 "Self-Supervised Learning", "Neural Architecture Search"]
PAPER_ADJS = ["Novel", "Robust", "Scalable", "Adaptive", "Unified", "Hierarchical", "Dynamic"]
PAPER_APPS = ["E-commerce", "Social Networks", "Healthcare", "Autonomous Driving",
              "Smart Cities", "Financial Risk", "Industrial IoT", "Search Engines"]
CONF_LOCATIONS = ["Vancouver, Canada", "New Orleans, USA", "Vienna, Austria",
                  "Seoul, South Korea", "Singapore", "Barcelona, Spain",
                  "Sydney, Australia", "Tokyo, Japan", "Paris, France",
                  "San Francisco, USA", "London, UK", "Beijing, China"]

# ── 多样化人设类型（决定教育路径与全日制/非全日制等）────────────────────────
# 权重越大越常见；十分体现人才库真实分布（主体为普通全日制，其余为多样补充）
PERSONA_TYPES = [
    ("normal_fulltime", 30),      # 普通全日制（本/硕/博）
    ("in_school", 12),            # 在读（未毕业，应届/实习）
    ("part_time", 10),            # 非全日制（在职考研/成人本科等）
    ("zhuanshengben", 10),        # 专升本（专科→本科）
    ("zhuanshengben_grad", 8),    # 专升本后读研
    ("gap_then_grad", 8),         # 本科毕业gap一两年再读研
    ("overseas_master", 6),       # 海外硕士（国内本科+海外硕士）
    ("career_switch", 6),         # 转行（非计算机本科→转做技术）
]


def _weighted_persona() -> str:
    """按权重随机抽取一种人设类型。"""
    population = [p for p, _ in PERSONA_TYPES]
    weights = [w for _, w in PERSONA_TYPES]
    return random.choices(population, weights=weights, k=1)[0]


# ── 留学专属人设类型（study_abroad 模式专用，覆盖各种留学路径）────────────────
# 1) 国内本科 + 海外硕士（含"水硕"1年制）
# 2) 国内本科 + 直接海外读博
# 3) 本硕均在海外
# 4) 海外本科 + 国内硕士
# 5) 国内本科 + 海外硕士 + 海外博士（连续深造）
# 6) 海外本科 + 海外硕士 + 国内博士（回国深造）
ABROAD_PERSONA_TYPES = [
    ("dom_bach_overseas_master", 34),     # 国内本科 + 海外硕士（含水硕）
    ("dom_bach_overseas_phd", 14),        # 国内本科 + 直接海外读博
    ("overseas_bach_master", 18),         # 本硕均在海外
    ("overseas_bach_dom_master", 16),     # 海外本科 + 国内硕士
    ("dom_bach_overseas_master_phd", 10), # 国内本科 + 海外硕士 + 海外博士
    ("overseas_bach_master_dom_phd", 8),  # 海外本硕 + 国内博士
]


def _weighted_abroad_persona() -> str:
    population = [p for p, _ in ABROAD_PERSONA_TYPES]
    weights = [w for _, w in ABROAD_PERSONA_TYPES]
    return random.choices(population, weights=weights, k=1)[0]

# ── LLM 写作 Prompt ──────────────────────────────────────────────────────────
RESUME_WRITER_SYSTEM = """你是一名资深的技术简历撰写专家。请根据给定的候选人画像骨架，撰写一份**真实、详尽、自然**的中文技术简历。

写作要求：
1. 输出**纯文本简历正文**，不要使用 JSON、Markdown 表格或代码块，就像真人写在简历里的样子。
2. 篇幅按骨架中「目标字数」控制，信息密度高，避免空话套话。字数少的简历应相应精简项目与工作经历的数量和篇幅，字数多的则更详尽。
3. 必须包含以下板块（用自然的小标题或分段呈现即可）：个人信息、教育经历、专业技能、获奖与证书、工作/实习经历、项目经历。
4. **教育经历**必须严格按骨架给定的「教育路径」逐段如实呈现：可能包含专科、专升本、非全日制、在读、海外院校、gap 间隔等情况。
   - 非全日制要明确写出「（非全日制）」；全日制可不特别标注或标注「（全日制）」。
   - 在读状态要写明「至今/在读」，且不要编造尚未取得的学位。
   - 专升本要体现「专科→本科」两段经历；gap 经历要让本科毕业时间与读研入学时间之间留出 1~2 年空档（可在工作经历里体现这段时间）。
   - 海外院校名称要严格保留骨架给定的「英文(中文)」格式（例如 University of Oxford(牛津大学)），不要改写、不要只写中文或只写英文；可在备注里自然体现 GPA、雅思/托福、海外科研/实习等留学相关细节，写出真实留学求职者的风格。
5. **专业技能**部分分点详细罗列，体现对数据库、大数据生态、编程语言、机器学习/深度学习算法、大模型/Agent 技术栈的掌握程度（精通/熟练/熟悉/了解）。技能水平要与人设和年限匹配，应届/在读者不宜样样精通。
6. **工作经历**每段写明公司、城市、职位、起止时间，并用一段话描述具体职责与产出；在读/应届者可只写实习经历或留空。
7. **项目经历**是重点：每个项目要有名称、时间、角色，并用要点详细描述技术方案、所用框架/算法、关键参数、工程实现与最终结果，技术细节要具体（例如模型结构、数据处理、部署方式、调优手段、评估指标等）。
8. 时间线要自洽合理，与画像骨架给定的学历、年限、公司、教育路径完全一致，不得自相矛盾。
9. 直接输出简历正文，不要任何额外说明、前言或结语。"""

RESUME_WRITER_USER = """请根据以下候选人画像骨架撰写完整简历正文：

{skeleton}

请严格按写作要求输出一份约 {target_chars} 字（汉字计）的真实风格简历正文，并严格遵循给定的教育路径与全日制/非全日制标注。"""


class ResumeGeneratorSkill(BaseSkill):
    """合成简历生成Skill（文本优先：先写长文本简历，再交给提取 agent 入库）"""

    def __init__(self):
        super().__init__(name="resume_generator", description="生成自然语言长文本合成简历并提取入库")
        self._resume_dir = Path(SYNTHETIC_DIR) / "resumes"
        self._resume_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """生成合成简历

        params:
            count: 生成份数（默认 100，上限 1000）
            use_llm: 是否用 LLM 写作（默认 True，False 则用本地模板长文本）
            save_to_db: 是否提取入库（默认 True）
            save_text: 是否把简历原文落盘（默认 True）
            concurrency: 并发度（默认 1=串行；>1 时并发生成，加速但对 LLM/DB 压力更大）
            id_offset: 文件命名/姓名编号起始偏移（追加场景下避免覆盖已有文件）
            study_abroad: 是否生成留学专属简历（默认 False；True 时用 ABROAD_PERSONA_TYPES）
        """
        count = min(int(params.get("count", 100)), 1000)
        use_llm = params.get("use_llm", True)
        save_to_db = params.get("save_to_db", True)
        save_text = params.get("save_text", True)
        concurrency = max(1, int(params.get("concurrency", 1)))
        id_offset = int(params.get("id_offset", 0))
        study_abroad = bool(params.get("study_abroad", False))

        # 延迟导入，避免循环依赖
        extractor = None
        if save_to_db:
            from backend.skills.resume_extraction_skill import ResumeExtractionSkill
            extractor = ResumeExtractionSkill()

        generated = 0
        extracted = 0
        candidate_ids: List[int] = []
        samples: List[Dict[str, Any]] = []

        sem = asyncio.Semaphore(concurrency)

        async def _produce_one(i: int) -> Dict[str, Any]:
            """生成单份：写作 -> 落盘 -> 提取入库+向量化。受信号量限流。"""
            async with sem:
                skeleton = self._build_skeleton(i + id_offset, study_abroad=study_abroad)
                # Step 1: 生成一大段自然语言简历文本
                resume_text = await self._write_resume_text(skeleton, use_llm)

                # Step 2: 落盘原文
                text_path = None
                if save_text:
                    text_path = self._resume_dir / f"resume_{i+1+id_offset:04d}_{skeleton['name']}.txt"
                    try:
                        text_path.write_text(resume_text, encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"写入简历原文失败: {e}")

                # Step 3: 提取结构化字段并入库 + 向量化
                cid = None
                if extractor is not None:
                    try:
                        extract_result = await extractor.extract_and_index(resume_text=resume_text)
                        if extract_result.get("success"):
                            cid = extract_result.get("candidate_id")
                        else:
                            logger.warning(f"第{i+1}份简历提取失败: {extract_result.get('error')}")
                    except Exception as e:
                        logger.error(f"第{i+1}份简历提取异常: {e}")

                return {
                    "idx": i,
                    "name": skeleton["name"],
                    "candidate_id": cid,
                    "text_length": len(resume_text),
                    "text_path": str(text_path) if text_path else None,
                    "resume_text_preview": resume_text[:300] + ("..." if len(resume_text) > 300 else ""),
                }

        # 并发或串行执行
        if concurrency > 1:
            tasks = [asyncio.create_task(_produce_one(i)) for i in range(count)]
            done = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            done = []
            for i in range(count):
                try:
                    done.append(await _produce_one(i))
                except Exception as e:
                    done.append(e)

        # 汇总结果
        for item in done:
            if isinstance(item, Exception):
                logger.error(f"生成任务异常: {item}")
                continue
            generated += 1
            if item.get("candidate_id"):
                candidate_ids.append(item["candidate_id"])
                extracted += 1
            if item.get("idx", 99) < 3:
                samples.append({k: v for k, v in item.items() if k != "idx"})

        candidate_ids.sort()

        result: Dict[str, Any] = {
            "generated": generated,           # 生成的简历文本份数
            "extracted_to_db": extracted,     # 成功提取入库的份数
            "candidate_ids": candidate_ids,
            "resume_dir": str(self._resume_dir),
            "samples": samples,
            "method": "llm" if use_llm else "template",
            "concurrency": concurrency,
        }

        # 兼容旧返回字段
        if save_to_db:
            try:
                from backend.database.models import hr_db
                result["total_in_db"] = hr_db.get_all_candidates_count()
            except Exception:
                pass

        return result

    # ── 画像骨架 ──────────────────────────────────────────────────────────
    @staticmethod
    def _pick_school(tier: str) -> str:
        """按层级抽学校：985/211/双一流/普通本科/专科。"""
        pools = {
            "985": SCHOOLS_985,
            "211": SCHOOLS_211,
            "双一流": SCHOOLS_SHUANG,
            "普通本科": SCHOOLS_NORMAL,
            "专科": SCHOOLS_ZHUANKE,
        }
        return random.choice(pools.get(tier, SCHOOLS_NORMAL))

    @staticmethod
    def _pick_overseas(tier: str = None) -> str:
        """挑选一所海外院校（英文(中文) display 形式）。

        tier 指定等价层级时从对应池抽取（985=顶尖名校；211/双一流=普通海外名校/水硕高频）；
        不指定则全池抽取。
        """
        if tier and tier in OVERSEAS_POOLS:
            return random.choice(OVERSEAS_POOLS[tier])
        return random.choice(OVERSEAS_ALL)

    def _build_abroad_education_path(self, persona: str) -> Dict[str, Any]:
        """构造留学路径教育经历。海外院校用 英文(中文) 形式，国内院校用中文名。

        覆盖：国内本科+海外硕士(水硕)、国内本科+海外读博、本硕均海外、
        海外本科+国内硕士、国内本科+海外硕博、海外本硕+国内博士。
        """
        major = random.choice(MAJORS)
        stages: List[Dict[str, Any]] = []

        def dom_stage(degree, tier, start, end, note="", maj=None):
            return {
                "degree": degree, "school": self._pick_school(tier),
                "school_tier_hint": tier, "major": maj or major,
                "start": start, "end": end, "is_fulltime": True, "note": note,
            }

        def os_stage(degree, start, end, note="", maj=None, tier_pref=None):
            return {
                "degree": degree, "school": self._pick_overseas(tier_pref),
                "school_tier_hint": "海外名校（按QS/US News排名等价国内层级）",
                "major": maj or major, "start": start, "end": end,
                "is_fulltime": True, "note": note, "is_overseas": True,
            }

        if persona == "dom_bach_overseas_master":
            # 国内本科 + 海外硕士（常见1~1.5年水硕，也有2年）
            b_start = random.randint(2016, 2021)
            stages.append(dom_stage("本科", random.choice(["985", "211", "双一流", "普通本科"]),
                                    f"{b_start}.09", f"{b_start+4}.06"))
            m_start = b_start + 4
            m_len = random.choice([1, 1, 1, 2])  # 多为1年制
            m_end_y, m_end_m = (m_start + 1, "09") if m_len == 1 else (m_start + 2, "06")
            note = "海外授课型硕士（1年制）" if m_len == 1 else "海外研究型硕士（2年制）"
            stages.append(os_stage("硕士", f"{m_start}.09", f"{m_end_y}.{m_end_m}", note=note,
                                   tier_pref=random.choice(["985", "211", "双一流"])))
            highest, wy = "硕士", max(0, 2025 - m_end_y)
            status = random.choice(JOB_STATUS)

        elif persona == "dom_bach_overseas_phd":
            # 国内本科 + 直接海外读博（硕博连读或直博）
            b_start = random.randint(2014, 2019)
            stages.append(dom_stage("本科", random.choice(["985", "211", "双一流"]),
                                    f"{b_start}.09", f"{b_start+4}.06"))
            phd_start = b_start + 4
            phd_len = random.choice([4, 5])
            phd_done = random.random() < 0.6
            phd_end = f"{phd_start+phd_len}.06" if phd_done else "至今（在读）"
            stages.append(os_stage("博士", f"{phd_start}.09", phd_end,
                                   note="本科直接申请海外博士（直博）",
                                   tier_pref=random.choice(["985", "211"])))
            highest = "博士"
            wy = max(0, 2025 - (phd_start + phd_len)) if phd_done else 0
            status = random.choice(JOB_STATUS) if phd_done else random.choice(["应届", "在职看机会"])

        elif persona == "overseas_bach_master":
            # 本硕均在海外
            b_start = random.randint(2015, 2020)
            stages.append(os_stage("本科", f"{b_start}.09", f"{b_start+random.choice([3,4])}.06",
                                   note="海外本科", tier_pref=random.choice(["985", "211", "双一流"])))
            m_start = b_start + 4
            m_len = random.choice([1, 2])
            m_end = m_start + m_len
            stages.append(os_stage("硕士", f"{m_start}.09", f"{m_end}.06",
                                   note="海外硕士（本硕连读海外）",
                                   tier_pref=random.choice(["985", "211", "双一流"])))
            highest, wy = "硕士", max(0, 2025 - m_end)
            status = random.choice(JOB_STATUS)

        elif persona == "overseas_bach_dom_master":
            # 海外本科 + 国内硕士（回国读研）
            b_start = random.randint(2015, 2020)
            stages.append(os_stage("本科", f"{b_start}.09", f"{b_start+random.choice([3,4])}.06",
                                   note="海外本科", tier_pref=random.choice(["985", "211", "双一流"])))
            m_start = b_start + 4
            stages.append(dom_stage("硕士", random.choice(["985", "211", "双一流"]),
                                    f"{m_start}.09", f"{m_start+3}.06", note="海外本科回国读研"))
            highest, wy = "硕士", max(0, 2025 - (m_start + 3))
            status = random.choice(JOB_STATUS)

        elif persona == "dom_bach_overseas_master_phd":
            # 国内本科 + 海外硕士 + 海外博士
            b_start = random.randint(2012, 2017)
            stages.append(dom_stage("本科", random.choice(["985", "211", "双一流"]),
                                    f"{b_start}.09", f"{b_start+4}.06"))
            m_start = b_start + 4
            stages.append(os_stage("硕士", f"{m_start}.09", f"{m_start+1}.09",
                                   note="海外硕士", tier_pref=random.choice(["985", "211"])))
            phd_start = m_start + 2
            phd_done = random.random() < 0.6
            phd_end = f"{phd_start+4}.06" if phd_done else "至今（在读）"
            stages.append(os_stage("博士", f"{phd_start}.09", phd_end,
                                   note="海外博士", tier_pref=random.choice(["985", "211"])))
            highest = "博士"
            wy = max(0, 2025 - (phd_start + 4)) if phd_done else 0
            status = random.choice(JOB_STATUS) if phd_done else random.choice(["应届", "在职看机会"])

        else:  # overseas_bach_master_dom_phd：海外本硕 + 国内博士
            b_start = random.randint(2012, 2017)
            stages.append(os_stage("本科", f"{b_start}.09", f"{b_start+4}.06",
                                   note="海外本科", tier_pref=random.choice(["985", "211", "双一流"])))
            m_start = b_start + 4
            stages.append(os_stage("硕士", f"{m_start}.09", f"{m_start+1}.09",
                                   note="海外硕士", tier_pref=random.choice(["985", "211"])))
            phd_start = m_start + 2
            stages.append(dom_stage("博士", random.choice(["985", "211"]),
                                    f"{phd_start}.09", f"{phd_start+4}.06", note="海外本硕回国读博"))
            highest, wy = "博士", max(0, 2025 - (phd_start + 4))
            status = random.choice(JOB_STATUS)

        return {
            "stages": stages,
            "highest_education": highest,
            "work_years": wy,
            "job_status": status,
        }

    def _build_education_path(self, persona: str) -> Dict[str, Any]:
        """根据人设类型构造多段教育经历（含全日制/非全日制、在读、专升本、gap 等）。

        返回 {stages: [...], highest_education, current_year, work_years_base, job_status_hint}
        每个 stage: {degree, school, school_tier_hint, major, start, end, is_fulltime, note}
        """
        major = random.choice(MAJORS)
        non_cs_major = random.choice(["机械工程", "电气工程", "应用数学", "金融学", "自动化", "通信工程"])
        stages: List[Dict[str, Any]] = []

        def stage(degree, tier, start, end, fulltime=True, note="", maj=None):
            return {
                "degree": degree,
                "school": self._pick_school(tier),
                "school_tier_hint": tier,
                "major": maj or major,
                "start": start,
                "end": end,
                "is_fulltime": fulltime,
                "note": note,
            }

        if persona == "in_school":
            # 在读：本科或硕士尚未毕业
            if random.random() < 0.6:
                start = random.randint(2022, 2024)
                stages.append(stage("本科", random.choice(["985", "211", "双一流", "普通本科"]),
                                    f"{start}.09", "至今（在读）", note="本科在读"))
                highest, status, wy = "本科", "应届", 0
            else:
                b_start = random.randint(2019, 2021)
                stages.append(stage("本科", random.choice(["211", "双一流", "普通本科"]),
                                    f"{b_start}.09", f"{b_start+4}.06"))
                m_start = b_start + 4
                stages.append(stage("硕士", random.choice(["985", "211"]),
                                    f"{m_start}.09", "至今（在读）", note="硕士在读"))
                highest, status, wy = "硕士", "应届", 0

        elif persona == "part_time":
            # 非全日制：先工作，再在职读非全日制本科/硕士
            b_start = random.randint(2014, 2018)
            stages.append(stage("本科", "专科", f"{b_start}.09", f"{b_start+3}.06", maj=major) if random.random() < 0.4
                          else stage("本科", random.choice(["普通本科", "双一流"]), f"{b_start}.09", f"{b_start+4}.06"))
            pt_start = b_start + 6
            stages.append(stage("硕士", random.choice(["211", "普通本科", "双一流"]),
                               f"{pt_start}.09", f"{pt_start+3}.06", fulltime=False, note="非全日制在职研究生"))
            highest, status = "硕士", random.choice(["在职看机会", "在职不看"])
            wy = max(2, 2025 - (b_start + 4))

        elif persona == "zhuanshengben":
            # 专升本：专科→本科
            z_start = random.randint(2017, 2020)
            stages.append(stage("大专", "专科", f"{z_start}.09", f"{z_start+3}.06"))
            stages.append(stage("本科", random.choice(["普通本科", "双一流"]),
                               f"{z_start+3}.09", f"{z_start+5}.06", note="专升本"))
            highest, status, wy = "本科", random.choice(JOB_STATUS), max(0, 2025 - (z_start + 5))

        elif persona == "zhuanshengben_grad":
            # 专升本后读研
            z_start = random.randint(2015, 2018)
            stages.append(stage("大专", "专科", f"{z_start}.09", f"{z_start+3}.06"))
            stages.append(stage("本科", random.choice(["普通本科", "双一流"]),
                               f"{z_start+3}.09", f"{z_start+5}.06", note="专升本"))
            stages.append(stage("硕士", random.choice(["985", "211", "双一流"]),
                               f"{z_start+5}.09", f"{z_start+8}.06"))
            highest, status, wy = "硕士", random.choice(JOB_STATUS), max(0, 2025 - (z_start + 8))

        elif persona == "gap_then_grad":
            # 本科毕业gap一两年再读研
            b_start = random.randint(2015, 2019)
            gap = random.choice([1, 2])
            stages.append(stage("本科", random.choice(["211", "双一流", "普通本科"]),
                               f"{b_start}.09", f"{b_start+4}.06"))
            m_start = b_start + 4 + gap
            stages.append(stage("硕士", random.choice(["985", "211"]),
                               f"{m_start}.09", f"{m_start+3}.06",
                               note=f"本科毕业后gap{gap}年再读研"))
            highest, status, wy = "硕士", random.choice(JOB_STATUS), max(0, 2025 - (m_start + 3)) + gap

        elif persona == "overseas_master":
            # 海外硕士
            b_start = random.randint(2015, 2019)
            stages.append(stage("本科", random.choice(["985", "211", "双一流"]),
                               f"{b_start}.09", f"{b_start+4}.06"))
            m_start = b_start + 4
            overseas = random.choice(["新加坡国立大学", "南洋理工大学", "香港科技大学",
                                       "爱丁堡大学", "墨尔本大学", "南加州大学"])
            os_stage = stage("硕士", "985", f"{m_start}.09", f"{m_start+1}.12")
            os_stage["school"] = overseas
            os_stage["school_tier_hint"] = "海外（按一流大学对待，但不在国内名单内）"
            os_stage["note"] = "海外硕士"
            stages.append(os_stage)
            highest, status, wy = "硕士", random.choice(JOB_STATUS), max(0, 2025 - (m_start + 2))

        elif persona == "career_switch":
            # 转行：非计算机本科→转做技术
            b_start = random.randint(2014, 2018)
            stages.append(stage("本科", random.choice(["211", "双一流", "普通本科"]),
                               f"{b_start}.09", f"{b_start+4}.06", maj=non_cs_major))
            if random.random() < 0.5:
                stages.append(stage("硕士", random.choice(["985", "211", "双一流"]),
                                   f"{b_start+4}.09", f"{b_start+7}.06", note="跨专业转入计算机/软件方向"))
                highest, wy = "硕士", max(1, 2025 - (b_start + 7))
            else:
                highest, wy = "本科", max(1, 2025 - (b_start + 4))
            status = random.choice(["在职看机会", "离职", "在职不看"])

        else:  # normal_fulltime
            edu = random.choices(["博士", "硕士", "本科", "大专"], weights=[8, 35, 50, 7], k=1)[0]
            b_start = random.randint(2010, 2020)
            if edu == "大专":
                stages.append(stage("大专", "专科", f"{b_start}.09", f"{b_start+3}.06"))
                highest, wy = "大专", max(0, 2025 - (b_start + 3))
            else:
                b_tier = random.choices(["985", "211", "双一流", "普通本科"], weights=[25, 30, 20, 25], k=1)[0]
                stages.append(stage("本科", b_tier, f"{b_start}.09", f"{b_start+4}.06"))
                cur = b_start + 4
                if edu in ("硕士", "博士"):
                    stages.append(stage("硕士", random.choice(["985", "211", "双一流"]),
                                       f"{cur}.09", f"{cur+3}.06"))
                    cur += 3
                if edu == "博士":
                    stages.append(stage("博士", random.choice(["985", "211"]),
                                       f"{cur}.09", f"{cur+4}.06"))
                    cur += 4
                highest, wy = edu, max(0, 2025 - cur)
            status = random.choice(JOB_STATUS)

        return {
            "stages": stages,
            "highest_education": highest,
            "work_years": wy,
            "job_status": status,
        }

    def _build_skeleton(self, idx: int, study_abroad: bool = False) -> Dict[str, Any]:
        """随机生成候选人画像骨架，作为 LLM 写作的输入约束（含多样化人设）

        study_abroad=True 时走留学专属人设与教育路径。
        """
        if study_abroad:
            persona = _weighted_abroad_persona()
            edu_path = self._build_abroad_education_path(persona)
        else:
            persona = _weighted_persona()
            edu_path = self._build_education_path(persona)
        highest = edu_path["highest_education"]
        work_years = edu_path["work_years"]
        status = edu_path["job_status"]

        direction = random.choice(DIRECTIONS)
        age = 22 + work_years + (2 if highest == "硕士" else 4 if highest == "博士" else 0) + random.randint(0, 3)

        # 技能数量与年限/学历相关：在读/应届少一些
        is_student = status in ("应届", "实习") or work_years == 0
        skills = random.sample(TECH_SKILLS, random.randint(5, 8) if is_student else random.randint(8, 14))
        companies = [] if is_student and random.random() < 0.4 else random.sample(COMPANIES, random.randint(1, 3))
        awards = random.sample(AWARD_POOL, random.randint(1, 4))
        num_projects = random.randint(1, 2) if is_student else random.randint(2, 4)

        # 弹性目标字数 2000~8000（在读/应届偏短，资深偏长）
        if is_student:
            target_chars = random.randint(2000, 4000)
        elif work_years >= 6:
            target_chars = random.randint(5000, 8000)
        else:
            target_chars = random.randint(3000, 6000)

        # 现居地/家乡：留学人设有概率在海外
        if study_abroad and random.random() < 0.3:
            current_city = random.choice(OVERSEAS_CITY_LOCATIONS)
        else:
            current_city = random.choice(CITY_LOCATIONS)
        hometown = random.choice(HOMETOWN_LOCATIONS)

        # 论文发表与国际会议（硕士/博士/资深研究者有概率生成）
        publications = []
        conferences = []
        if highest in ("博士", "硕士") or (work_years >= 3 and random.random() < 0.2):
            publications = self._generate_publications(highest, work_years, is_student)
            conferences = self._generate_conferences(highest, work_years, is_student)

        # 动态扩展属性（随机生成，模拟真实简历中的多样化信息）
        extra_attributes = self._generate_extra_attributes(highest, is_student, age)

        return {
            "name": f"候选人_{idx+1:04d}",
            "persona": persona,
            "gender": random.choice(["男", "女"]),
            "age": age,
            "phone": f"1{random.randint(30,99)}{random.randint(10000000,99999999)}",
            "email": f"candidate_{idx+1}@example.com",
            "city": random.choice(CITIES),
            "current_city": current_city,
            "hometown": hometown,
            "highest_education": highest,
            "education_path": edu_path["stages"],
            "major": edu_path["stages"][-1]["major"] if edu_path["stages"] else random.choice(MAJORS),
            "direction": direction,
            "work_years": work_years,
            "current_position": "实习生" if is_student else random.choice(POSITIONS),
            "expected_salary": round(random.uniform(8, 80) * 1000, 0),
            "job_status": status,
            "skills": skills,
            "companies": companies,
            "awards": awards,
            "num_projects": num_projects,
            "target_chars": target_chars,
            "publications": publications,
            "conferences": conferences,
            "extra_attributes": extra_attributes,
        }

    # ── 论文 & 会议生成 ─────────────────────────────────────────────────────
    def _generate_publications(self, highest_edu: str, work_years: int, is_student: bool) -> List[Dict[str, Any]]:
        """根据学历和工作年限生成论文发表记录。"""
        # 确定论文数量
        if highest_edu == "博士":
            num_pubs = random.randint(3, 8)
        elif highest_edu == "硕士":
            num_pubs = random.randint(1, 4)
        else:
            num_pubs = random.randint(0, 2)

        publications = []
        current_year = 2025
        for _ in range(num_pubs):
            # 选择期刊或会议论文
            venue_type = random.choice(["journal", "conference"])
            # 根据学历分配等级概率
            if highest_edu == "博士":
                rank_weights = [0.3, 0.4, 0.3]  # A, B, C
            elif highest_edu == "硕士":
                rank_weights = [0.1, 0.4, 0.5]
            else:
                rank_weights = [0.05, 0.25, 0.7]

            rank_choice = random.choices(["CCF-A", "CCF-B", "CCF-C"], weights=rank_weights, k=1)[0]

            if venue_type == "journal":
                if rank_choice == "CCF-A":
                    pool = JOURNALS_CCF_A
                elif rank_choice == "CCF-B":
                    pool = JOURNALS_CCF_B
                else:
                    pool = JOURNALS_CCF_C
            else:
                if rank_choice == "CCF-A":
                    pool = CONFERENCES_CCF_A
                elif rank_choice == "CCF-B":
                    pool = CONFERENCES_CCF_B
                else:
                    pool = CONFERENCES_CCF_C

            if not pool:
                continue

            venue_info = random.choice(pool)
            venue_name = venue_info["name"]
            venue_rank = venue_info["rank"]

            # SCI 分区（仅期刊）
            sci_zone = ""
            if venue_type == "journal":
                if rank_choice == "CCF-A":
                    sci_zone = random.choice(["Q1", "Q1", "Q2"])
                elif rank_choice == "CCF-B":
                    sci_zone = random.choice(["Q1", "Q2", "Q2", "Q3"])
                else:
                    sci_zone = random.choice(["Q2", "Q3", "Q3", "Q4"])

            # 生成论文标题
            template = random.choice(PAPER_TITLE_TEMPLATES)
            title = template.format(
                adj=random.choice(PAPER_ADJS),
                method=random.choice(PAPER_METHODS),
                topic=random.choice(PAPER_TOPICS),
                app=random.choice(PAPER_APPS),
            )

            # 作者位次
            author_position = random.choices(
                ["first_author", "corresponding_author", "co_author"],
                weights=[0.4, 0.2, 0.4],
                k=1,
            )[0]

            # 发表年份
            if is_student:
                year = random.randint(current_year - 3, current_year)
            else:
                year = random.randint(current_year - work_years - 2, current_year)
                year = max(year, 2015)

            publications.append({
                "title": title,
                "venue": venue_name,
                "venue_type": venue_type,
                "venue_rank": venue_rank,
                "sci_zone": sci_zone,
                "year": year,
                "author_position": author_position,
                "doi": f"10.{random.randint(1000,9999)}/{random.randint(100000,999999)}",
            })

        return publications

    def _generate_conferences(self, highest_edu: str, work_years: int, is_student: bool) -> List[Dict[str, Any]]:
        """根据学历和工作年限生成国际会议参与记录。"""
        if highest_edu == "博士":
            num_confs = random.randint(2, 6)
        elif highest_edu == "硕士":
            num_confs = random.randint(1, 3)
        else:
            num_confs = random.randint(0, 2)

        conferences = []
        current_year = 2025
        roles = ["oral_presentation", "poster", "workshop", "attendee", "session_chair", "keynote"]
        role_weights_phd = [0.3, 0.25, 0.15, 0.15, 0.1, 0.05]
        role_weights_master = [0.2, 0.3, 0.2, 0.25, 0.05, 0.0]
        role_weights_other = [0.1, 0.2, 0.2, 0.4, 0.1, 0.0]

        for _ in range(num_confs):
            # 等级选择
            if highest_edu == "博士":
                rank_weights = [0.35, 0.4, 0.25]
            elif highest_edu == "硕士":
                rank_weights = [0.15, 0.4, 0.45]
            else:
                rank_weights = [0.05, 0.3, 0.65]

            rank_choice = random.choices(["CCF-A", "CCF-B", "CCF-C"], weights=rank_weights, k=1)[0]

            if rank_choice == "CCF-A":
                pool = CONFERENCES_CCF_A
            elif rank_choice == "CCF-B":
                pool = CONFERENCES_CCF_B
            else:
                pool = CONFERENCES_CCF_C

            if not pool:
                continue

            conf_info = random.choice(pool)

            # 角色
            if highest_edu == "博士":
                role = random.choices(roles, weights=role_weights_phd, k=1)[0]
            elif highest_edu == "硕士":
                role = random.choices(roles, weights=role_weights_master, k=1)[0]
            else:
                role = random.choices(roles, weights=role_weights_other, k=1)[0]

            # 年份
            if is_student:
                year = random.randint(current_year - 3, current_year)
            else:
                year = random.randint(current_year - work_years - 1, current_year)
                year = max(year, 2016)

            # 地点
            location = random.choice(CONF_LOCATIONS)

            # 如果是 oral/poster，生成论文标题
            paper_title = ""
            if role in ("oral_presentation", "poster"):
                template = random.choice(PAPER_TITLE_TEMPLATES)
                paper_title = template.format(
                    adj=random.choice(PAPER_ADJS),
                    method=random.choice(PAPER_METHODS),
                    topic=random.choice(PAPER_TOPICS),
                    app=random.choice(PAPER_APPS),
                )

            conferences.append({
                "conference_name": conf_info["name"],
                "conference_rank": conf_info["rank"],
                "year": year,
                "location": location,
                "role": role,
                "paper_title": paper_title,
            })

        return conferences

    @staticmethod
    def _generate_extra_attributes(highest_edu: str, is_student: bool, age: int) -> Dict[str, Any]:
        """随机生成动态扩展属性（GPA、爱好、目标岗位、民族、身高体重等）。

        并非所有简历都包含这些信息，通过概率控制模拟真实分布。
        """
        attrs: Dict[str, Any] = {}

        # GPA（学生/应届更常写）
        if is_student or random.random() < 0.3:
            gpa_style = random.choice(["4.0", "5.0", "100"])
            if gpa_style == "4.0":
                gpa = round(random.uniform(2.8, 4.0), 2)
                attrs["gpa"] = f"{gpa}/4.0"
            elif gpa_style == "5.0":
                gpa = round(random.uniform(3.5, 5.0), 2)
                attrs["gpa"] = f"{gpa}/5.0"
            else:
                gpa = random.randint(72, 98)
                attrs["gpa"] = f"{gpa}/100"

        # 爱好（约40%的简历会写）
        if random.random() < 0.4:
            hobby_pool = ["篮球", "足球", "游泳", "跑步", "健身", "羽毛球", "乒乓球",
                          "阅读", "摄影", "旅行", "音乐", "吉他", "钢琴", "绘画",
                          "编程", "开源贡献", "写博客", "电竞", "桌游", "烹饪",
                          "登山", "骑行", "瑜伽", "滑雪", "潜水"]
            hobbies = random.sample(hobby_pool, random.randint(2, 5))
            attrs["hobbies"] = ",".join(hobbies)

        # 目标岗位（约50%会写）
        if random.random() < 0.5:
            target_pool = ["后端开发工程师", "Java高级工程师", "算法工程师",
                           "数据开发工程师", "全栈工程师", "架构师",
                           "大数据开发", "AI工程师", "NLP算法工程师",
                           "推荐算法工程师", "搜索工程师", "数据分析师"]
            attrs["target_job"] = random.choice(target_pool)

        # 民族（约15%会写）
        if random.random() < 0.15:
            ethnicity_pool = ["汉族", "汉族", "汉族", "汉族", "汉族",
                              "回族", "满族", "壮族", "苗族", "维吾尔族", "藏族", "蒙古族"]
            attrs["ethnicity"] = random.choice(ethnicity_pool)

        # 身高体重（约10%会写）
        if random.random() < 0.10:
            attrs["height_cm"] = random.randint(155, 190)
            attrs["weight_kg"] = random.randint(45, 95)

        # 婚姻状况（年龄大的更可能写）
        if age >= 26 and random.random() < 0.15:
            attrs["marital_status"] = random.choices(
                ["未婚", "已婚", "离异"], weights=[0.5, 0.45, 0.05], k=1)[0]

        # 政治面貌（约20%会写）
        if random.random() < 0.20:
            attrs["political_status"] = random.choices(
                ["群众", "共青团员", "中共党员"], weights=[0.3, 0.4, 0.3], k=1)[0]

        # 语言能力（约35%会写）
        if random.random() < 0.35:
            lang_pool = ["英语CET-4", "英语CET-6", "英语CET-6(560+)", "英语雅思7.0",
                         "英语托福100+", "日语N2", "日语N1", "德语B1", "法语A2"]
            langs = random.sample(lang_pool, random.randint(1, 2))
            attrs["languages"] = ",".join(langs)

        # 自我评价（约30%会写）
        if random.random() < 0.30:
            eval_pool = [
                "具有良好的团队协作能力和沟通能力，善于解决复杂技术问题",
                "热爱技术，持续学习，有较强的自驱力和责任心",
                "注重代码质量，熟悉敏捷开发流程，有良好的工程素养",
                "逻辑思维强，善于抽象建模，对分布式系统有深入理解",
                "具备全栈视野，能独立负责从需求分析到上线的完整流程",
                "对AI/大模型技术充满热情，持续关注前沿论文和开源项目",
            ]
            attrs["self_evaluation"] = random.choice(eval_pool)

        return attrs

    @staticmethod
    def _format_education_path(stages: List[Dict[str, Any]]) -> str:
        """把多段教育经历渲染成 prompt 文本（逐段标注层级、专业、全日制与备注）。"""
        if not stages:
            return "（无）"
        lines = []
        for st in stages:
            ft = "全日制" if st.get("is_fulltime", True) else "非全日制"
            tier = st.get("school_tier_hint", "")
            note = st.get("note", "")
            note_str = f"，备注：{note}" if note else ""
            lines.append(
                f"  · {st.get('start','')}-{st.get('end','')}，{st.get('school','')}"
                f"（{tier}），{st.get('degree','')}，{st.get('major','')}，{ft}{note_str}"
            )
        return "\n".join(lines)

    def _skeleton_to_prompt(self, s: Dict[str, Any]) -> str:
        companies = s.get("companies") or []
        companies_str = "、".join(companies) if companies else "（暂无正式工作经历，可写实习或留空）"
        lines = [
            f"- 姓名：{s['name']}",
            f"- 人设类型（仅供你把握风格，不要写进简历）：{s.get('persona','')}",
            f"- 性别/年龄：{s['gender']}，{s['age']}岁",
            f"- 联系方式：电话 {s['phone']}，邮箱 {s['email']}",
            f"- 所在城市：{s['city']}",
            f"- 现居地：{s.get('current_city', '')}（必须写入简历个人信息中，格式为'省份/城市'或'国家/城市'）",
            f"- 家乡/籍贯：{s.get('hometown', '')}（必须写入简历个人信息中，格式为'省份/城市'）",
            f"- 最高学历：{s['highest_education']}",
            f"- 教育路径（必须逐段如实呈现，注意全日制/非全日制与时间空档）：\n{self._format_education_path(s.get('education_path', []))}",
            f"- 技术方向：{s['direction']}",
            f"- 工作年限：{s['work_years']}年",
            f"- 当前/目标职位：{s['current_position']}，求职状态：{s['job_status']}",
            f"- 期望薪资：约{int(s['expected_salary'])}元/月",
            f"- 需覆盖的技术栈：{ '、'.join(s['skills']) }",
            f"- 工作/实习过的公司：{companies_str}",
            f"- 获奖与证书：{ '、'.join(s['awards']) }",
            f"- 需要撰写 {s['num_projects']} 个详细的项目经历（每个项目 4~7 条技术要点）",
            f"- 目标字数：约 {s.get('target_chars', 4000)} 字",
        ]
        # 论文发表
        pubs = s.get("publications", [])
        if pubs:
            lines.append("- 论文发表：")
            for p in pubs:
                sci_str = f"，SCI {p['sci_zone']}" if p.get("sci_zone") else ""
                pos_map = {"first_author": "第一作者", "corresponding_author": "通讯作者", "co_author": "合作者"}
                lines.append(
                    f"  · [{p['venue_rank']}{sci_str}] {p['title']} - {p['venue']} ({p['year']})，{pos_map.get(p['author_position'], p['author_position'])}"
                )
        # 国际会议
        confs = s.get("conferences", [])
        if confs:
            lines.append("- 国际会议参与：")
            role_map = {
                "oral_presentation": "口头报告",
                "poster": "海报展示",
                "workshop": "研讨会",
                "attendee": "参会者",
                "session_chair": "分会主席",
                "keynote": "特邀报告",
            }
            for c in confs:
                paper_str = f"，论文：{c['paper_title']}" if c.get("paper_title") else ""
                lines.append(
                    f"  · [{c['conference_rank']}] {c['conference_name']} ({c['year']}，{c['location']})，角色：{role_map.get(c['role'], c['role'])}{paper_str}"
                )
        # 动态扩展属性（需自然融入简历正文中）
        extra = s.get("extra_attributes", {})
        if extra:
            extra_parts = []
            if extra.get("gpa"):
                extra_parts.append(f"GPA: {extra['gpa']}")
            if extra.get("hobbies"):
                extra_parts.append(f"兴趣爱好: {extra['hobbies']}")
            if extra.get("target_job"):
                extra_parts.append(f"目标岗位: {extra['target_job']}")
            if extra.get("ethnicity"):
                extra_parts.append(f"民族: {extra['ethnicity']}")
            if extra.get("height_cm"):
                extra_parts.append(f"身高: {extra['height_cm']}cm")
            if extra.get("weight_kg"):
                extra_parts.append(f"体重: {extra['weight_kg']}kg")
            if extra.get("marital_status"):
                extra_parts.append(f"婚姻状况: {extra['marital_status']}")
            if extra.get("political_status"):
                extra_parts.append(f"政治面貌: {extra['political_status']}")
            if extra.get("languages"):
                extra_parts.append(f"语言能力: {extra['languages']}")
            if extra.get("self_evaluation"):
                extra_parts.append(f"自我评价: {extra['self_evaluation']}")
            if extra_parts:
                lines.append(f"- 其他个人信息（请自然融入简历正文中）：{'；'.join(extra_parts)}")
        return "\n".join(lines)

    # ── 简历文本生成 ──────────────────────────────────────────────────────
    async def _write_resume_text(self, skeleton: Dict[str, Any], use_llm: bool) -> str:
        if use_llm:
            try:
                text = await self._llm_write(skeleton)
                if text and len(text) > 800:
                    return text
                logger.warning("LLM 生成文本过短，降级为模板")
            except Exception as e:
                logger.error(f"LLM 写作失败，降级为模板: {e}")
        return self._template_write(skeleton)

    async def _llm_write(self, skeleton: Dict[str, Any]) -> str:
        from backend.models.longcat_client import chat_completion

        system = RESUME_WRITER_SYSTEM
        target_chars = skeleton.get("target_chars", 4000)
        user = RESUME_WRITER_USER.format(
            skeleton=self._skeleton_to_prompt(skeleton),
            target_chars=target_chars,
        )

        # 根据目标字数动态估算 max_tokens（粗略：1 汉字 ≈ 1.6 token，再留余量）
        max_tokens = min(8192, max(2048, int(target_chars * 2.0)))

        # chat_completion 是同步调用，放到线程池避免阻塞事件循环
        msg = await asyncio.to_thread(
            chat_completion, system, user, None, 0.75, max_tokens
        )
        return (msg.content or "").strip()

    # ── 本地模板长文本（降级方案，无需 LLM）────────────────────────────────
    def _template_write(self, s: Dict[str, Any]) -> str:
        skills = s["skills"]
        parts: List[str] = []
        parts.append(f"{s['name']}")
        parts.append(f"性别：{s['gender']}    年龄：{s['age']}岁    目标：{s['current_position']}")
        parts.append(f"电话：{s['phone']}    邮箱：{s['email']}    现居地：{s.get('current_city', s['city'])}    家乡：{s.get('hometown', '')}    求职状态：{s['job_status']}")
        parts.append("")
        parts.append("【教育经历】")
        for st in s.get("education_path", []):
            ft = "全日制" if st.get("is_fulltime", True) else "非全日制"
            note = f"  [{st['note']}]" if st.get("note") else ""
            parts.append(
                f"{st.get('start','')}-{st.get('end','')}  {st.get('school','')}"
                f"（{st.get('degree','')}/{ft}）  {st.get('major','')}{note}"
            )
        parts.append("")
        parts.append("【专业技能】")
        bucket = lambda n: random.sample(skills, min(n, len(skills)))
        parts.append(f"1、熟悉关系型与非关系型数据库（{ '、'.join(bucket(3)) }），具备 SQL 开发、存储过程编写与性能优化经验；")
        parts.append(f"2、熟悉大数据生态（{ '、'.join(bucket(4)) }），具备离线与实时数据处理能力；")
        parts.append(f"3、熟练使用 { '、'.join(bucket(3)) } 等编程语言/框架进行工程开发；")
        parts.append("4、熟悉 SVM、随机森林、XGBoost、LightGBM、GCN 等机器学习算法，以及 Transformer、LSTM、R-CNN 系列深度学习模型；")
        parts.append("5、掌握 LangChain、LangGraph，熟悉 RAG、记忆存储、多智能体 Agent、harness 等大模型应用范式；")
        parts.append("6、熟悉 Docker、ElasticSearch，会使用 Git，能用 Flask/FastAPI 搭建算法接口并进行联调测试。")
        parts.append("")
        parts.append("【获奖与证书】")
        for a in s["awards"]:
            parts.append(f"- {a}")
        parts.append("")
        parts.append("【工作/实习经历】")
        year = 2025 - s["work_years"]
        comp_list = s.get("companies") or []
        for ci, comp in enumerate(comp_list):
            start = year + ci
            end = "至今" if ci == len(comp_list) - 1 else f"{start+1}.06"
            parts.append(f"{start}.07-{end}  {comp}（{s['city']}）  {random.choice(POSITIONS)}")
            parts.append(f"负责{ s['direction'] }相关工作，参与核心系统的设计、开发与性能优化，"
                         f"基于 { '、'.join(bucket(3)) } 完成数据处理、特征工程与模型/服务落地，"
                         f"推动需求从方案设计到上线交付的全流程实现，并持续迭代优化效果。")
        parts.append("")
        parts.append("【项目经历】")
        proj_names = ["智能推荐系统", "实时数据入库平台", "图像语义分割模型", "企业智能客服系统",
                      "离职预测模型", "网页自动爬虫", "人才画像匹配检索"]
        for pi in range(s["num_projects"]):
            pname = random.choice(proj_names)
            parts.append(f"{2024+pi}.0{random.randint(1,9)}  {pname}（{random.choice(['技术负责人','核心开发','算法开发'])}）")
            techs = bucket(4)
            parts.append(f"1、基于 { '、'.join(techs) } 搭建整体技术方案，完成数据采集、清洗与建模的端到端链路；")
            parts.append(f"2、使用 { random.choice(skills) } 实现核心模块，针对关键性能瓶颈进行针对性优化；")
            parts.append("3、设计并实现了完整的特征体系与处理流程，保证数据质量与计算正确性；")
            parts.append(f"4、引入 { random.choice(skills) } 进行模型训练与调优，完成交叉验证与超参数搜索；")
            parts.append("5、对结果进行可解释性分析与评估，输出专项分析报告支撑业务决策；")
            parts.append("6、负责工程化部署与接口封装，完成功能测试、联调与上线迭代。")
            parts.append("")

        # 论文发表
        pubs = s.get("publications", [])
        if pubs:
            parts.append("【论文发表】")
            pos_map = {"first_author": "第一作者", "corresponding_author": "通讯作者", "co_author": "合作者"}
            for p in pubs:
                sci_str = f"  SCI {p['sci_zone']}" if p.get("sci_zone") else ""
                parts.append(
                    f"- [{p['venue_rank']}{sci_str}] {p['title']}"
                )
                parts.append(
                    f"  发表于 {p['venue']}，{p['year']}年，{pos_map.get(p['author_position'], p['author_position'])}，DOI: {p.get('doi', '')}"
                )
            parts.append("")

        # 国际会议
        confs = s.get("conferences", [])
        if confs:
            parts.append("【国际会议】")
            role_map = {
                "oral_presentation": "口头报告",
                "poster": "海报展示",
                "workshop": "研讨会",
                "attendee": "参会者",
                "session_chair": "分会主席",
                "keynote": "特邀报告",
            }
            for c in confs:
                paper_str = f"  论文：{c['paper_title']}" if c.get("paper_title") else ""
                parts.append(
                    f"- [{c['conference_rank']}] {c['conference_name']}，{c['year']}年，{c['location']}"
                )
                parts.append(
                    f"  角色：{role_map.get(c['role'], c['role'])}{paper_str}"
                )
            parts.append("")

        # 动态扩展属性
        extra = s.get("extra_attributes", {})
        if extra:
            extra_lines = []
            if extra.get("gpa"):
                extra_lines.append(f"GPA：{extra['gpa']}")
            if extra.get("ethnicity"):
                extra_lines.append(f"民族：{extra['ethnicity']}")
            if extra.get("political_status"):
                extra_lines.append(f"政治面貌：{extra['political_status']}")
            if extra.get("height_cm"):
                extra_lines.append(f"身高：{extra['height_cm']}cm")
            if extra.get("weight_kg"):
                extra_lines.append(f"体重：{extra['weight_kg']}kg")
            if extra.get("marital_status"):
                extra_lines.append(f"婚姻状况：{extra['marital_status']}")
            if extra.get("languages"):
                extra_lines.append(f"语言能力：{extra['languages']}")
            if extra.get("target_job"):
                extra_lines.append(f"目标岗位：{extra['target_job']}")
            if extra.get("hobbies"):
                extra_lines.append(f"兴趣爱好：{extra['hobbies']}")
            if extra.get("self_evaluation"):
                extra_lines.append(f"自我评价：{extra['self_evaluation']}")
            if extra_lines:
                parts.append("【其他信息】")
                for line in extra_lines:
                    parts.append(f"- {line}")
                parts.append("")

        return "\n".join(parts)
