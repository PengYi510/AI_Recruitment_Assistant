"""RAG检索Skill - BM25+BGE-M3三路召回 + LLM动态约束提取 + 硬约束预过滤 + JD长文本智能预处理"""
import logging, re, math, json
from typing import Dict, Any, List, Optional, Set
from collections import Counter
from pathlib import Path

import jieba
from backend.skills.base_skill import BaseSkill
from backend.models.multimodal_fusion import multimodal_fusion
from backend.vector_db.client import vector_db
from backend.database.models import hr_db
from backend.config import RAG_TOP_K, RAG_BM25_WEIGHT, RAG_DENSE_WEIGHT
from backend.models.longcat_client import chat_json
from backend.utils.candidate_category import (
    get_grad_year_filter_for_query,
    get_fresh_grad_year_range,
    get_intern_grad_year_min,
)

logger = logging.getLogger(__name__)

# ── 海外院校排名知识库（用于 QS/US News 排名约束过滤）──────────────────────
_OVERSEAS_KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "overseas_school_rankings.json"


def _load_overseas_ranking() -> List[Dict[str, Any]]:
    """加载海外院校排名数据，返回完整列表"""
    try:
        data = json.loads(_OVERSEAS_KB_PATH.read_text(encoding="utf-8"))
        return data.get("universities", [])
    except Exception as e:
        logger.error(f"[RAG] 海外院校排名知识库加载失败: {e}")
        return []


# 全局加载一次
_OVERSEAS_UNIVERSITIES = _load_overseas_ranking()


def _get_schools_by_rank(max_rank: int, ranking_type: str = "best") -> List[str]:
    """根据排名阈值获取院校名称列表（用于SQL LIKE匹配）

    ranking_type:
        "qs" - 仅看 QS 排名
        "usnews" - 仅看 US News 排名
        "best" - 取两个排名中更优的（默认）
    """
    schools = []
    for uni in _OVERSEAS_UNIVERSITIES:
        if ranking_type == "qs":
            rank = uni.get("qs_2025_rank") or 9999
        elif ranking_type == "usnews":
            rank = uni.get("usnews_2025_rank") or 9999
        else:  # best
            rank = uni.get("best_rank") or 9999
        if isinstance(rank, (int, float)) and rank <= max_rank:
            # 收集多种名称形式用于匹配
            if uni.get("name_en"):
                schools.append(uni["name_en"])
            if uni.get("name_cn"):
                schools.append(uni["name_cn"])
    return schools


class RAGRetrievalSkill(BaseSkill):
    """RAG检索Skill: BM25稀疏+BGE-M3稠密+加权融合三路召回 + 硬约束预过滤"""

    # JD长文本检测阈值（超过此字数认为是完整JD而非简短查询）
    _JD_LENGTH_THRESHOLD = 100

    def __init__(self):
        super().__init__(name="rag_retrieval", description="三路召回检索候选人")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        top_k = params.get("top_k", RAG_TOP_K)

        # Step 0: JD长文本预处理 — 将完整JD提炼为精准检索query
        is_long_jd = len(query) > self._JD_LENGTH_THRESHOLD
        if is_long_jd:
            search_query = self._extract_search_query_from_jd(query)
            logger.info(f"[RAG] JD长文本检测: 原文{len(query)}字 → 检索query: '{search_query}'")
        else:
            search_query = query

        # Step 0.5: 从查询中提取硬约束和软约束（使用原始完整query以提取更多信息）
        constraints = self._extract_constraints(query)
        soft_constraints = constraints.pop("_soft_constraints", {})
        logger.info(f"[RAG] 硬约束提取: {constraints}")
        if soft_constraints:
            logger.info(f"[RAG] 软约束提取: {soft_constraints}")

        # Step 1: 如果有硬约束，先做SQL预过滤获取满足条件的候选人ID集合
        # 注意：软约束不参与 SQL 预过滤，仅参与后续评分加权
        # constraint_ids: None=无约束, 非空Set=有匹配, 空Set=有约束但无人满足
        hard_constraints_only = {k: v for k, v in constraints.items() if not k.startswith("_")}
        constraint_ids = self._sql_prefilter(constraints) if hard_constraints_only else None

        # Step 1.5: 硬约束交集为空 → 直接返回空结果（不再退化为全量检索）
        if hard_constraints_only and constraint_ids is not None and len(constraint_ids) == 0:
            logger.warning(f"[RAG] 所有硬约束交集为空，无候选人满足全部条件: {constraints}")
            return {
                "candidates": [],
                "total_found": 0,
                "retrieval_method": "bm25+dense_fusion",
                "constraints_detected": constraints,
                "soft_constraints_detected": soft_constraints,
                "constraint_matched_count": 0,
                "no_match_reason": "数据库中没有同时满足所有硬约束条件的候选人",
            }

        # Step 2: BM25稀疏检索（扩大召回范围，使用精炼后的search_query）
        bm25_top_k = max(top_k * 3, 60)  # BM25阶段多召回一些
        bm25_results = self._bm25_search(search_query, bm25_top_k)

        # Step 3: BGE-M3稠密检索（使用精炼后的search_query）
        dense_results = self._dense_search(search_query, top_k * 2)

        # Step 3.5: 确保硬约束候选人一定在候选池中（补充注入）
        # 注意：无论硬约束命中多少人都要注入——否则当满足硬约束的人未被
        # bm25/dense 自然召回时（例如约束命中 42/110 人但都不在向量 top_k 内），
        # 候选池里将没有任何满足约束的人，最终结果被下游按约束过滤为空。
        if constraint_ids:
            existing_ids = {r["candidate_id"] for r in bm25_results} | {r["candidate_id"] for r in dense_results}
            missing_ids = constraint_ids - existing_ids
            if missing_ids:
                self._ensure_cache_loaded()
                # 为缺失的约束候选人补算 BM25 相关度分，保证它们之间仍有区分度
                query_terms = self._expand_synonyms(self._tokenize(search_query))
                N = max(len(self._text_cache), 1)
                for cid in missing_ids:
                    if cid in self._data_cache:
                        doc_terms = self._text_cache.get(cid, [])
                        rel = self._compute_bm25(query_terms, doc_terms, N) if doc_terms else 0.0
                        bm25_results.append({
                            "candidate_id": cid,
                            "score": rel,
                            "data": self._data_cache[cid]
                        })

        # Step 4: 加权融合 + 硬约束优先 + 软约束加分
        soft_constraint_ids = None
        if soft_constraints:
            soft_constraint_ids = self._sql_prefilter_soft(soft_constraints)
            logger.info(f"[RAG] 软约束匹配: {len(soft_constraint_ids) if soft_constraint_ids else 0}人满足偏好条件")
        fused_results = self._fuse_results(bm25_results, dense_results, top_k * 2,
                                           constraint_ids, soft_constraint_ids)

        # Step 5: 最终截断
        final_results = fused_results[:top_k]

        return {
            "candidates": final_results,
            "total_found": len(final_results),
            "retrieval_method": "bm25+dense_fusion",
            "constraints_detected": constraints,
            "soft_constraints_detected": soft_constraints,
            "constraint_matched_count": len(constraint_ids) if constraint_ids else 0,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 硬约束提取与预过滤
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # JD长文本预处理（改进1）
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_search_query_from_jd(self, jd_text: str) -> str:
        """从完整JD文本中提取精炼的检索query
        
        策略：基于规则从JD中提取岗位名称+核心技能+关键要求，拼接为短查询。
        这比直接用500字JD去做BM25/Dense检索效果好得多。
        """
        parts = []

        # 1. 提取岗位名称（通常在第一行或第一句）
        title_match = re.search(
            r'([\u4e00-\u9fff\w]+(?:工程师|开发|架构师|分析师|科学家|专家|实习[生]?|经理|主管|运营|设计师|测试|运维))',
            jd_text[:100]
        )
        if title_match:
            parts.append(title_match.group(1))

        # 2. 提取岗位类型关键词
        position_keywords = [
            "数据挖掘", "机器学习", "深度学习", "自然语言处理", "NLP", "大模型",
            "算法", "AI", "人工智能", "计算机视觉", "CV", "推荐系统",
            "后端", "前端", "全栈", "移动端", "iOS", "Android",
            "数据分析", "数据开发", "数据科学", "数据工程",
            "测试", "运维", "DevOps", "产品", "运营", "UI",
        ]
        for kw in position_keywords:
            if kw in jd_text:
                if kw not in parts:
                    parts.append(kw)

        # 3. 提取技术栈关键词（从岗位要求部分）
        tech_keywords = [
            'Python', 'Java', 'C++', 'C/C++', 'Go', 'Rust', 'Scala',
            'JavaScript', 'TypeScript', 'React', 'Vue', 'Angular', 'Node.js',
            'SQL', 'MySQL', 'Hive', 'Spark', 'Hadoop', 'Flink', 'Kafka',
            'PyTorch', 'TensorFlow', 'Sklearn', 'scikit-learn',
            'Docker', 'Kubernetes', 'Redis', 'MongoDB', 'Elasticsearch',
            'Spring', 'SpringBoot', 'Django', 'FastAPI', 'Flask',
            'Linux', 'Git', 'AWS', 'GCP',
        ]
        jd_lower = jd_text.lower()
        found_techs = [kw for kw in tech_keywords if kw.lower() in jd_lower]
        # 只取前6个最重要的技能避免过长
        parts.extend(found_techs[:6])

        # 4. 提取实习/应届相关标识
        if re.search(r'实习|在校|应届|届[及以]', jd_text):
            parts.append("实习生")
        if re.search(r'(\d{4})届', jd_text):
            grad_match = re.search(r'(\d{4})届', jd_text)
            if grad_match:
                parts.append(f"{grad_match.group(1)}届")

        # 5. 提取学历要求
        if "博士" in jd_text:
            parts.append("博士")
        elif "硕士" in jd_text or "研究生" in jd_text:
            parts.append("硕士")
        elif "本科" in jd_text:
            parts.append("本科")

        # 6. 提取专业方向
        major_keywords = ["计算机", "软件工程", "信息管理", "统计学", "数学", "电子信息",
                         "人工智能", "数据科学", "自动化", "通信"]
        for major in major_keywords:
            if major in jd_text and major not in parts:
                parts.append(major)
                break  # 只取第一个匹配的专业

        # 拼接为精炼query
        if parts:
            search_query = " ".join(parts)
            logger.info(f"[RAG] JD预处理提取关键词: {parts}")
            return search_query

        # 兜底：截取前80字
        return jd_text[:80]

    # ══════════════════════════════════════════════════════════════════════════
    # LLM 动态约束提取（核心创新：任意新属性无需改代码即可自动识别）
    # ══════════════════════════════════════════════════════════════════════════

    # LLM 约束提取的 System Prompt（模板部分，运行时会注入动态 attr_key 列表）
    _LLM_CONSTRAINT_SYSTEM_PROMPT_TEMPLATE = """你是一个招聘查询约束提取器。从用户的自然语言查询中提取所有筛选条件。

你必须返回一个严格的 JSON 对象，包含以下三部分：

1. "standard_constraints": 硬性约束（用户明确要求、必须满足的条件），可能包含：
   - "school": 具体学校名（如"华中科技大学"）
   - "school_tier": 院校层级（"985"/"211"/"双一流"）
   - "overseas_rank": 海外排名上限数字（如 50 表示 QS前50）
   - "overseas_rank_type": 排名类型（"qs"/"usnews"/"best"）
   - "min_work_years": 最低工作年限（整数）
   - "max_work_years": 最高工作年限（整数）
   - "highest_education": 最低学历要求（"本科"/"硕士"/"博士"）
   - "required_skills": 技术技能列表（如 ["React", "Node.js", "Python"]）
   - "location": 工作地点
   - "gender": 性别（"男"/"女"）
   - "min_pub_rank": 最低论文等级（"CCF-A"/"CCF-B"/"CCF-C"）
   - "has_publications": 是否要求有论文（布尔值）
   - "is_intern": 是否实习生岗位（布尔值）
   - "is_fresh_grad": 是否应届生（布尔值）
   - "require_fulltime": 是否要求全日制（布尔值）

2. "extra_constraints": 动态扩展属性约束（对应 candidate_extra_attributes 表的 key-value）。
   数据库中当前存在的属性维度有：{available_extra_keys}
   你必须使用上述列表中的精确 key 名（如使用 "height_cm" 而非 "height"）。

   每个 extra_constraint 的格式为 "key": {{"operator": "...", "value": ...}}
   示例：
   - "gpa": {{"operator": ">=", "value": 3.5}}  — GPA约束
   - "target_job": {{"operator": "contains", "value": "后端"}}  — 目标岗位
   - "hobbies": {{"operator": "contains", "value": "篮球"}}  — 爱好
   - "languages": {{"operator": "contains", "value": "英语"}}  — 语言能力
   - "height_cm": {{"operator": ">=", "value": 170}}  — 身高(厘米)
   - "weight_kg": {{"operator": "<=", "value": 70}}  — 体重(千克)
   - "ethnicity": {{"operator": "==", "value": "汉族"}}  — 民族
   - 以及上述列表中的任何其他属性...

   每个 extra_constraint 的 operator 可以是：
   - ">=": 大于等于（数值比较）
   - "<=": 小于等于（数值比较）
   - ">": 大于
   - "<": 小于
   - "==": 精确等于
   - "contains": 包含（字符串模糊匹配）

3. "soft_constraints": 软性偏好约束（用户表达为"优先/偏好/最好/尽量/加分项"的条件，不是硬性要求）。
   格式与 standard_constraints 相同，但这些条件不应作为过滤条件，而是作为加分项。
   典型的软性表述词：优先、偏好、最好、尽量、加分项、优先考虑、倾向于、更好。

重要规则：
- 只提取用户明确表达的约束，不要推测或添加用户未提及的条件
- 如果某个字段用户没有提及，就不要包含在结果中
- 数值类型的值用数字表示，不要用字符串
- 如果查询中没有任何约束，返回空的 standard_constraints、extra_constraints 和 soft_constraints
- 注意区分"目标岗位"（target_job，候选人期望的岗位）和查询中的岗位描述（用于技能匹配）
- extra_constraints 的 key 必须严格使用数据库中实际存在的 key 名（见上方列表）
- 【关键】区分硬性要求与软性偏好：
  - "要求985" / "必须985" / "985院校" → standard_constraints（硬性）
  - "优先985" / "985优先" / "最好是985" / "偏好985" → soft_constraints（软性）
  - "有xx经验加分" / "优先考虑有xx的" → soft_constraints（软性）

示例输入："帮我找一个华中科技大学毕业的前端开发，会React和Node.js，8年左右经验，目标岗位DevOps工程师，GPA 3以上的"
示例输出：
{{
  "standard_constraints": {{
    "school": "华中科技大学",
    "min_work_years": 7,
    "required_skills": ["React", "Node.js"]
  }},
  "extra_constraints": {{
    "gpa": {{"operator": ">=", "value": 3.0}},
    "target_job": {{"operator": "contains", "value": "DevOps"}}
  }},
  "soft_constraints": {{}}
}}

示例输入："找一个应届生，要求身高160以上，女性"
示例输出：
{{
  "standard_constraints": {{
    "is_fresh_grad": true,
    "gender": "女"
  }},
  "extra_constraints": {{
    "height_cm": {{"operator": ">=", "value": 160}}
  }},
  "soft_constraints": {{}}
}}

示例输入："帮我找商业分析实习生，优先985/211院校，最好有数据分析经验"
示例输出：
{{
  "standard_constraints": {{
    "is_intern": true
  }},
  "extra_constraints": {{}},
  "soft_constraints": {{
    "school_tier": "985",
    "required_skills": ["数据分析"]
  }}
}}"""

    def _build_constraint_prompt(self) -> str:
        """动态构建约束提取 Prompt，注入数据库中实际存在的 attr_key 列表。
        
        这确保 LLM 输出的 key 名与数据库一致，避免 key 不匹配被丢弃。
        """
        known_keys = self._get_known_extra_attr_keys()
        keys_str = ", ".join(sorted(known_keys)) if known_keys else "（暂无动态属性）"
        return self._LLM_CONSTRAINT_SYSTEM_PROMPT_TEMPLATE.format(
            available_extra_keys=keys_str
        )

    def _llm_extract_constraints(self, query: str) -> Dict[str, Any]:
        """使用 LLM 从自然语言中动态提取任意约束条件
        
        返回格式：
        {
            "standard_constraints": {...},  # 硬性标准字段约束
            "extra_constraints": {...},     # 动态扩展属性约束
            "soft_constraints": {...},      # 软性偏好约束（优先/最好等）
        }
        
        失败时返回空字典，由调用方 fallback 到正则提取。
        """
        try:
            system_prompt = self._build_constraint_prompt()
            result = chat_json(
                system=system_prompt,
                user=f"请从以下查询中提取所有筛选约束：\n\n{query}",
                temperature=0.05,
            )
            
            standard = result.get("standard_constraints", {})
            extra = result.get("extra_constraints", {})
            soft = result.get("soft_constraints", {})
            
            # 基本校验：确保返回的是字典
            if not isinstance(standard, dict):
                standard = {}
            if not isinstance(extra, dict):
                extra = {}
            if not isinstance(soft, dict):
                soft = {}
            
            logger.info(f"[RAG] LLM约束提取成功: standard={list(standard.keys())}, "
                        f"extra={list(extra.keys())}, soft={list(soft.keys())}")
            return {"standard_constraints": standard, "extra_constraints": extra,
                    "soft_constraints": soft}
            
        except Exception as e:
            logger.warning(f"[RAG] LLM约束提取失败({type(e).__name__}: {e})，将fallback到正则提取")
            return {}

    # ══════════════════════════════════════════════════════════════════════════
    # 硬约束提取与预过滤（正则兜底 + LLM增强）
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """从自然语言查询中提取硬约束和软约束（LLM优先 + 正则兜底）
        
        策略：
        1. 先调用 LLM 提取约束（能识别任意新属性如 GPA、target_job 等）
        2. 同时用正则提取作为兜底（保证 LLM 不可用时系统仍可工作）
        3. 合并两者结果：LLM 结果优先，正则结果补充
        4. 正则兜底：对含"优先/偏好/最好/尽量"修饰的约束，移入软约束
        
        返回字典中：
        - 普通 key（如 school_tier, min_work_years）= 硬约束
        - "_soft_constraints" key = 软约束字典（不参与 SQL 过滤，仅用于评分加权）
        - "_extra_constraints" key = 动态扩展属性约束
        """
        # ── Phase 1: LLM 动态提取 ──
        llm_result = self._llm_extract_constraints(query)
        llm_standard = llm_result.get("standard_constraints", {})
        llm_extra = llm_result.get("extra_constraints", {})
        llm_soft = llm_result.get("soft_constraints", {})
        
        # ── Phase 2: 正则兜底提取（原有逻辑） ──
        constraints = self._regex_extract_constraints(query)
        
        # ── Phase 2.5: 正则层面的软约束检测 ──
        # 如果 LLM 没有返回 soft_constraints，用正则检测"优先/偏好/最好"修饰的约束
        soft_keys_from_regex = set()
        if not llm_soft:
            soft_keys_from_regex = self._detect_soft_constraint_keys(query, constraints)
        
        # ── Phase 3: 合并策略 ──
        # LLM 提取的标准约束覆盖正则结果（LLM 更准确）
        if llm_standard:
            for key, value in llm_standard.items():
                if value is not None and value != "" and value != []:
                    constraints[key] = value
        
        # LLM 提取的动态扩展属性约束存入特殊字段
        # 但需剔除数据集中并不存在的 attr_key（LLM 幻觉属性，如把"海外留学"
        # 编成 overseas_experience）——这类约束会被 _filter_by_extra_attributes
        # 跳过，但在此处提前剔除可让日志/调试更清晰，且避免误导下游。
        # 增强：对于不在白名单中的 key，尝试模糊匹配数据库实际 key（兜底纠错）。
        if llm_extra:
            valid_keys = self._get_known_extra_attr_keys()
            filtered_extra = {}
            for k, v in llm_extra.items():
                if k in valid_keys:
                    filtered_extra[k] = v
                else:
                    # 模糊匹配兜底：如 LLM 输出 "height" 而数据库是 "height_cm"
                    matched_key = self._fuzzy_match_attr_key(k, valid_keys)
                    if matched_key:
                        logger.info(
                            f"[RAG] 动态属性 key 自动纠正: '{k}' → '{matched_key}'"
                        )
                        filtered_extra[matched_key] = v
                    else:
                        logger.info(
                            f"[RAG] 忽略数据集中不存在的动态属性约束 '{k}'"
                            f"（已交由标准约束路径处理，如 has_overseas_edu）"
                        )
            if filtered_extra:
                constraints["_extra_constraints"] = filtered_extra

        # ── Phase 4: 构建软约束字典 ──
        soft_constraints = {}
        
        # 4a. 从 LLM 的 soft_constraints 合并
        if llm_soft:
            for key, value in llm_soft.items():
                if value is not None and value != "" and value != []:
                    soft_constraints[key] = value
        
        # 4b. 从正则检测到的软约束 key 中，将对应约束从硬约束移到软约束
        for sk in soft_keys_from_regex:
            if sk in constraints and sk not in ("_extra_constraints", "_soft_constraints"):
                soft_constraints[sk] = constraints.pop(sk)
        
        # 4c. 如果 LLM 返回了 soft_constraints，确保这些 key 不在硬约束中
        if llm_soft:
            for key in llm_soft:
                if key in constraints and key not in ("_extra_constraints", "_soft_constraints"):
                    constraints.pop(key)
        
        if soft_constraints:
            constraints["_soft_constraints"] = soft_constraints
            logger.info(f"[RAG] 软约束识别: {list(soft_constraints.keys())}")
            
        return constraints

    # 数据集中真实存在的 extra attr_key 缓存（避免每次查库）
    _known_extra_attr_keys: Optional[Set[str]] = None

    def _get_known_extra_attr_keys(self) -> Set[str]:
        """返回 candidate_extra_attributes 表中真实存在的 attr_key 集合（带缓存）。"""
        if RAGRetrievalSkill._known_extra_attr_keys is None:
            try:
                with hr_db._get_conn() as conn:
                    rows = conn.execute(
                        "SELECT DISTINCT attr_key FROM candidate_extra_attributes"
                    ).fetchall()
                RAGRetrievalSkill._known_extra_attr_keys = {r[0] for r in rows}
            except Exception as e:
                logger.warning(f"[RAG] 加载 extra attr_key 白名单失败: {e}")
                RAGRetrievalSkill._known_extra_attr_keys = set()
        return RAGRetrievalSkill._known_extra_attr_keys

    @staticmethod
    def _fuzzy_match_attr_key(key: str, valid_keys: Set[str]) -> Optional[str]:
        """尝试将 LLM 输出的 key 模糊匹配到数据库实际的 attr_key。
        
        策略（按优先级）：
        1. key 是某个 valid_key 的前缀（如 "height" → "height_cm"）
        2. valid_key 是 key 的前缀（如 "height_centimeters" → "height_cm"...不太常见）
        3. 去掉下划线和常见后缀后完全匹配
        
        只在有唯一匹配时返回，避免歧义。
        """
        if not valid_keys:
            return None
        
        # 策略1: key 是某个 valid_key 的前缀
        prefix_matches = [vk for vk in valid_keys if vk.startswith(key + "_") or vk.startswith(key)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        
        # 策略2: 去掉常见后缀/单位词后匹配
        # 例如 "height" 去匹配 "height_cm"；"weight" 去匹配 "weight_kg"
        suffixes_to_strip = ["_cm", "_kg", "_mm", "_m", "_years", "_months"]
        normalized_map = {}
        for vk in valid_keys:
            base = vk
            for suffix in suffixes_to_strip:
                if vk.endswith(suffix):
                    base = vk[:-len(suffix)]
                    break
            normalized_map[base] = vk
        
        if key in normalized_map:
            return normalized_map[key]
        
        # 策略3: 下划线替换为空后比较（如 "heightcm" vs "height_cm"）
        key_flat = key.replace("_", "").replace("-", "").lower()
        for vk in valid_keys:
            vk_flat = vk.replace("_", "").replace("-", "").lower()
            if key_flat == vk_flat:
                return vk
        
        return None

    def _regex_extract_constraints(self, query: str) -> Dict[str, Any]:
        """从自然语言查询中提取硬约束条件（纯正则方式，作为LLM的兜底）"""
        constraints = {}

        # 1. 提取院校约束 — 先匹配全称，再匹配简称（避免短简称误匹配）
        # 排除层级描述词被误识别为学校名（如"双一流大学"、"985大学"、"211大学"等）
        tier_keywords = {'985', '211', '双一流', '一流', 'C9', 'c9'}
        # 查询动词/量词等噪声前缀词：避免被贪婪并入学校名（如"找一个清华大学"→"清华大学"）
        noise_prefixes = (
            "找一个", "找一位", "找一名", "找个", "找位", "找名", "找",
            "想要", "需要", "想找", "要找", "查找", "查询", "推荐", "寻找", "搜索",
            "帮我找", "给我找", "帮我推荐", "有没有", "有一个", "有一位", "有",
            "一个", "一位", "一名", "某个", "某位",
        )

        def _strip_school_noise(name: str) -> str:
            """从匹配到的学校名里循环剥离前缀噪声词，直到不再变化。"""
            changed = True
            while changed:
                changed = False
                for p in noise_prefixes:
                    if name.startswith(p) and len(name) > len(p):
                        name = name[len(p):]
                        changed = True
                        break
            return name

        # 先尝试直接匹配全称（更精确）
        school_pattern = re.search(r'([\u4e00-\u9fff]{2,8}(?:大学|学院|科技大学|理工大学|师范大学))', query)
        if school_pattern:
            matched_school = _strip_school_noise(school_pattern.group(1))
            # 检查是否是层级描述词+大学/学院（如"双一流大学"）— 这不是具体学校名
            school_prefix = re.sub(r'(?:大学|学院|科技大学|理工大学|师范大学)$', '', matched_school)
            # 剥离后若过短（<2字核心名）则视为无效学校约束
            if school_prefix not in tier_keywords and len(matched_school) >= 3:
                constraints["school"] = matched_school
        else:
            # 简称匹配：按简称长度降序排列，优先匹配更长的简称（"西南交大"优先于"交大"）
            sorted_synonyms = sorted(self._SCHOOL_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True)
            for short_name, full_name in sorted_synonyms:
                if short_name in query:
                    constraints["school"] = full_name
                    break

        # 1.5 提取院校层级约束（985/211/双一流/C9）
        tier_patterns = [
            (r'[Cc]9|C9联盟|九校联盟', '985'),   # C9是985子集，用985来筛
            (r'985', '985'),
            (r'211', '211'),
            (r'双一流|"双一流"', '双一流'),
        ]
        for pattern, tier in tier_patterns:
            if re.search(pattern, query):
                constraints["school_tier"] = tier
                break

        # 1.6 提取海外院校排名约束（QS前50/QS Top100/US News前50 等）
        # 支持多种表达：QS前50、QS排名前50、QS Top 50、QS50强、世界前50、US News前100等
        ranking_patterns = [
            # QS排名
            (r'[Qq][Ss]\s*(?:排名|世界排名)?(?:前|[Tt]op)\s*(\d+)', 'qs'),
            (r'[Qq][Ss]\s*(\d+)\s*强', 'qs'),
            (r'[Qq][Ss]\s*(?:排名|世界排名)?\s*(\d+)\s*(?:以内|名以内)', 'qs'),
            # US News排名
            (r'[Uu]\.?[Ss]\.?\s*[Nn]ews\s*(?:排名)?(?:前|[Tt]op)\s*(\d+)', 'usnews'),
            (r'[Uu][Ss][Nn][Ee][Ww][Ss]\s*(?:排名)?(?:前|[Tt]op)\s*(\d+)', 'usnews'),
            # 通用"世界排名前X"/"全球前X"/"海外名校前X"
            (r'(?:世界|全球|国际)\s*(?:排名)?(?:前|[Tt]op)\s*(\d+)', 'best'),
            (r'(?:海外|留学|国外)\s*(?:名校|院校|高校)\s*(?:排名)?(?:前|[Tt]op)\s*(\d+)', 'best'),
            (r'(?:前|[Tt]op)\s*(\d+)\s*(?:名校|院校|高校|大学)', 'best'),
        ]
        for pattern, ranking_type in ranking_patterns:
            rank_match = re.search(pattern, query)
            if rank_match:
                max_rank = int(rank_match.group(1))
                if 1 <= max_rank <= 500:  # 合理范围
                    constraints["overseas_rank"] = max_rank
                    constraints["overseas_rank_type"] = ranking_type
                    logger.info(f"[RAG] 检测到海外排名约束: {ranking_type} Top {max_rank}")
                break

        # 1.7 检测"留学"/"海外"/"海归"等关键词（无具体排名时标记为海外经历约束）
        if "overseas_rank" not in constraints and not constraints.get("school"):
            if re.search(r'留学|海外|海归|出国|国外(?:院校|大学|学校)|归国', query):
                constraints["has_overseas_edu"] = True

        # 1.8 性别约束（正则兜底）
        if re.search(r'女性|女生|女', query) and not re.search(r'男', query):
            constraints["gender"] = "女"
        elif re.search(r'男性|男生|男', query) and not re.search(r'女', query):
            constraints["gender"] = "男"

        # 2. 提取工作年限约束
        # 支持多种表达："X年以上", "至少X年", "X+年经验", "X到Y年"(取X), "X年Java"(无修饰词也取)
        year_patterns = [
            r'(\d+)\s*年以上',               # X年以上
            r'至少\s*(\d+)\s*年',              # 至少X年
            r'(\d+)\+\s*年',                  # X+年
            r'(\d+)\s*到\s*\d+\s*年',           # X到Y年 (取最小X)
            r'(\d+)\s*年(?:经验|工作)',           # X年经验/X年工作
            r'(\d+)\s*年(?:[A-Za-z]|[一-鿿])',    # X年+其他内容（如"3年Java"）
        ]
        for pattern in year_patterns:
            year_match = re.search(pattern, query)
            if year_match:
                years = int(year_match.group(1))
                if years >= 1:  # 过滤掉无意义的数字
                    constraints["min_work_years"] = years
                break

        # 2.5 实习生/应届生场景检测（动态时间，不硬编码年份）
        category_filters = get_grad_year_filter_for_query(query)
        if category_filters:
            if category_filters.get("is_intern"):
                constraints["is_intern"] = True
                constraints["max_work_years"] = category_filters["max_work_years"]
            elif category_filters.get("is_fresh_grad"):
                constraints["is_fresh_grad"] = True
                constraints["max_work_years"] = category_filters["max_work_years"]
            if "grad_year_min" in category_filters:
                constraints["grad_year_min"] = category_filters["grad_year_min"]
            if "grad_year_max" in category_filters:
                constraints["grad_year_max"] = category_filters["grad_year_max"]

        # 3. 提取学历约束
        edu_patterns = {
            "博士": r'博士|PhD|博士生',
            "硕士": r'硕士|研究生|master',
            "本科": r'本科|学士|bachelor',
        }
        for edu, pattern in edu_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                constraints["highest_education"] = edu
                break

        # 3.5 全日制/非全日制偏好
        # 场景1：显式排除非全日制（如"不要非全日制"）
        # 场景2：正面要求全日制（如"全日制硕士"、"全日制本科"）
        if re.search(r"不要?非全日?制?|排除非全|非全日?制?除外|必须全日制|只要全日制|要求全日制|全日制[优優]先", query):
            constraints["require_fulltime"] = True
        elif re.search(r"全日制", query) and not re.search(r"非全日?制?[也亦]可|接受非全", query):
            # "全日制"出现在查询中且没有"接受非全"的表述 → 视为要求全日制
            constraints["require_fulltime"] = True

        # 4. 提取技能约束（精确匹配常见技术关键词）
        tech_keywords = ['Java', 'Python', 'Go', 'C++', 'JavaScript', 'TypeScript', 'React',
                         'Vue', 'Spring', 'Django', 'FastAPI', 'Docker', 'Kubernetes', 'MySQL',
                         'Redis', 'Kafka', 'Elasticsearch', 'TensorFlow', 'PyTorch', 'Spark',
                         'Node.js', 'Rust', 'Scala', 'Flutter', 'iOS', 'Android', 'AWS',
                         'GCP', 'Linux', 'Nginx', 'RabbitMQ', 'MongoDB', 'PostgreSQL', 'gRPC',
                         'Hive', 'Sklearn', 'Pandas', 'NumPy', 'SQL']
        query_lower = query.lower()
        found_skills = [kw for kw in tech_keywords if kw.lower() in query_lower]
        if found_skills:
            constraints["required_skills"] = found_skills

        # 5. 提取论文/期刊等级约束（CCF-A/B/C、SCI分区）
        # 支持表达："有CCF-A论文"、"发过CCF A类论文"、"SCI一区"、"顶会论文"等
        pub_rank_patterns = [
            (r'CCF[\s\-]?A|CCF\s*A类|顶刊|顶会论文|顶级期刊|顶级会议', 'CCF-A'),
            (r'CCF[\s\-]?B|CCF\s*B类', 'CCF-B'),
            (r'CCF[\s\-]?C|CCF\s*C类', 'CCF-C'),
        ]
        for pattern, rank in pub_rank_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                constraints["min_pub_rank"] = rank
                break

        # SCI分区约束
        sci_patterns = [
            (r'SCI\s*[一1]\s*区|SCI\s*Q1|一区论文', 'Q1'),
            (r'SCI\s*[二2]\s*区|SCI\s*Q2|二区论文', 'Q2'),
            (r'SCI\s*[三3]\s*区|SCI\s*Q3', 'Q3'),
            (r'SCI\s*[四4]\s*区|SCI\s*Q4', 'Q4'),
        ]
        for pattern, zone in sci_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                constraints["max_sci_zone"] = zone
                break

        # 通用论文/发表约束（无具体等级，仅要求有论文）
        if "min_pub_rank" not in constraints and "max_sci_zone" not in constraints:
            if re.search(r'有论文|发[过表]论文|论文发表|学术论文|发表过|有发表', query):
                constraints["has_publications"] = True

        # 6. 提取国际会议约束
        # 支持表达："参加过CCF-A会议"、"有顶会经历"、"国际会议经验"等
        conf_rank_patterns = [
            (r'(?:参加|参与|出席).*?CCF[\s\-]?A|CCF[\s\-]?A.*?会议|顶会经[历验]|顶级会议经[历验]', 'CCF-A'),
            (r'(?:参加|参与|出席).*?CCF[\s\-]?B|CCF[\s\-]?B.*?会议', 'CCF-B'),
            (r'(?:参加|参与|出席).*?CCF[\s\-]?C|CCF[\s\-]?C.*?会议', 'CCF-C'),
        ]
        for pattern, rank in conf_rank_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                constraints["min_conf_rank"] = rank
                break

        # 通用国际会议约束
        if "min_conf_rank" not in constraints:
            if re.search(r'国际会议|学术会议|参[加与]过会议|会议经[历验]', query):
                constraints["has_conferences"] = True

        # 7. 作者位次约束
        if re.search(r'第一作者|一作|first\s*author', query, re.IGNORECASE):
            constraints["author_position"] = "first_author"
        elif re.search(r'通讯作者|通讯|corresponding', query, re.IGNORECASE):
            constraints["author_position"] = "corresponding_author"

        return constraints

    @staticmethod
    def _detect_soft_constraint_keys(query: str, constraints: Dict[str, Any]) -> Set[str]:
        """正则检测哪些已提取的约束实际上是软性偏好（优先/最好/尽量等）。
        
        返回应被标记为软约束的 key 集合。
        
        策略：在查询文本中搜索"优先xxx"、"最好xxx"等模式，
        如果模式中提及了某个约束的值，则该约束为软约束。
        """
        soft_keys = set()
        
        # 软性修饰词正则
        soft_pattern = r'(?:优先|偏好|最好|尽量|加分|倾向|更好|优先考虑|首选)\s*(?:是)?'
        
        # 检测 school_tier 软约束（如 "优先985/211"）
        if "school_tier" in constraints:
            if re.search(soft_pattern + r'.*?(?:985|211|双一流)', query):
                soft_keys.add("school_tier")
            elif re.search(r'(?:985|211|双一流)\s*(?:优先|加分|更好)', query):
                soft_keys.add("school_tier")
        
        # 检测 school 软约束
        if "school" in constraints:
            school = constraints["school"]
            if re.search(soft_pattern + r'.*?' + re.escape(school), query):
                soft_keys.add("school")
            elif re.search(re.escape(school) + r'\s*(?:优先|加分|更好)', query):
                soft_keys.add("school")
        
        # 检测 highest_education 软约束
        if "highest_education" in constraints:
            if re.search(soft_pattern + r'.*?(?:本科|硕士|博士|研究生)', query):
                soft_keys.add("highest_education")
        
        # 检测 required_skills 软约束（如 "最好有xx经验"）
        if "required_skills" in constraints:
            if re.search(soft_pattern + r'.*?(?:经验|技能|能力|掌握)', query):
                soft_keys.add("required_skills")
            elif re.search(r'(?:经验|技能|能力).{0,4}(?:优先|加分)', query):
                soft_keys.add("required_skills")
        
        # 检测 has_publications 软约束
        if "has_publications" in constraints:
            if re.search(soft_pattern + r'.*?(?:论文|发表|发过)', query):
                soft_keys.add("has_publications")
        
        # 检测 min_work_years 软约束
        if "min_work_years" in constraints:
            if re.search(soft_pattern + r'.*?\d+\s*年', query):
                soft_keys.add("min_work_years")
        
        if soft_keys:
            logger.info(f"[RAG] 正则检测到软约束 keys: {soft_keys}")
        
        return soft_keys

    def _sql_prefilter(self, constraints: Dict[str, Any]) -> Set[int]:
        """基于硬约束在SQL层预过滤候选人，返回满足所有约束的ID集合"""
        candidate_ids = None

        # 学校约束：同时在 candidates.school 和 education_history 表中查
        if "school" in constraints:
            school = constraints["school"]
            with hr_db._get_conn() as conn:
                # 先查 education_history 子表
                rows1 = conn.execute(
                    "SELECT DISTINCT candidate_id FROM education_history WHERE school LIKE ?",
                    (f"%{school}%",)
                ).fetchall()
                # 也查 candidates 主表的 school 字段（兼容旧数据）
                rows2 = conn.execute(
                    "SELECT id FROM candidates WHERE school LIKE ?",
                    (f"%{school}%",)
                ).fetchall()
                school_ids = {r[0] for r in rows1} | {r[0] for r in rows2}
                candidate_ids = school_ids if candidate_ids is None else candidate_ids & school_ids

        # 院校层级约束（985/211/双一流）
        if "school_tier" in constraints:
            tier = constraints["school_tier"]
            with hr_db._get_conn() as conn:
                # 985包含所有985高校；211包含985+211；双一流包含985+211+双一流
                if tier == "985":
                    tier_condition = "school_tier = '985'"
                elif tier == "211":
                    tier_condition = "school_tier IN ('985', '211')"
                else:  # 双一流
                    tier_condition = "school_tier IN ('985', '211', '双一流')"
                rows = conn.execute(
                    f"SELECT DISTINCT candidate_id FROM education_history WHERE {tier_condition}"
                ).fetchall()
                tier_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 院校层级过滤({tier}): {len(tier_ids)}人满足")
                candidate_ids = tier_ids if candidate_ids is None else candidate_ids & tier_ids

        # 海外院校排名约束（QS前50/US News前100等）
        if "overseas_rank" in constraints:
            max_rank = constraints["overseas_rank"]
            ranking_type = constraints.get("overseas_rank_type", "best")
            target_schools = _get_schools_by_rank(max_rank, ranking_type)
            if target_schools:
                with hr_db._get_conn() as conn:
                    # 构建 LIKE 条件：任一院校名匹配即可
                    like_conditions = " OR ".join(["school LIKE ?" for _ in target_schools])
                    params = [f"%{s}%" for s in target_schools]
                    sql = f"SELECT DISTINCT candidate_id FROM education_history WHERE ({like_conditions})"
                    rows = conn.execute(sql, params).fetchall()
                    rank_ids = {r[0] for r in rows}
                    logger.info(f"[RAG] 海外排名过滤({ranking_type} Top{max_rank}): "
                                f"知识库{len(target_schools)}所院校, {len(rank_ids)}人满足")
                    candidate_ids = rank_ids if candidate_ids is None else candidate_ids & rank_ids
            else:
                logger.warning(f"[RAG] 海外排名知识库中无 {ranking_type} Top{max_rank} 的院校")
                candidate_ids = set()  # 无匹配

        # 海外教育经历约束（仅标记有留学经历，不限排名）
        if constraints.get("has_overseas_edu"):
            with hr_db._get_conn() as conn:
                # 海外院校特征：school 字段包含英文字符
                rows = conn.execute(
                    "SELECT DISTINCT candidate_id FROM education_history WHERE school GLOB '*[A-Za-z]*'"
                ).fetchall()
                overseas_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 海外教育经历过滤: {len(overseas_ids)}人有海外院校记录")
                candidate_ids = overseas_ids if candidate_ids is None else candidate_ids & overseas_ids

        # 性别约束
        if "gender" in constraints:
            gender = constraints["gender"]
            with hr_db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM candidates WHERE gender = ?",
                    (gender,)
                ).fetchall()
                gender_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 性别过滤({gender}): {len(gender_ids)}人满足")
                candidate_ids = gender_ids if candidate_ids is None else candidate_ids & gender_ids

        # 工作年限约束（最小年限）
        if "min_work_years" in constraints:
            min_years = constraints["min_work_years"]
            with hr_db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM candidates WHERE work_years >= ?",
                    (min_years,)
                ).fetchall()
                year_ids = {r[0] for r in rows}
                candidate_ids = year_ids if candidate_ids is None else candidate_ids & year_ids

        # 工作年限约束（最大年限 — 实习生/应届生场景）
        if "max_work_years" in constraints:
            max_years = constraints["max_work_years"]
            with hr_db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM candidates WHERE work_years <= ?",
                    (max_years,)
                ).fetchall()
                year_ids = {r[0] for r in rows}
                candidate_ids = year_ids if candidate_ids is None else candidate_ids & year_ids

        # 毕业年份约束（基于 education_history.end_date）
        grad_year_min = constraints.get("grad_year_min")
        grad_year_max = constraints.get("grad_year_max")
        if grad_year_min or (grad_year_max and grad_year_max < 9999):
            with hr_db._get_conn() as conn:
                # 查找最高学历的毕业时间在指定范围内的候选人
                conditions = []
                params = []
                if grad_year_min:
                    conditions.append("CAST(SUBSTR(end_date, 1, 4) AS INTEGER) >= ?")
                    params.append(grad_year_min)
                if grad_year_max and grad_year_max < 9999:
                    conditions.append("CAST(SUBSTR(end_date, 1, 4) AS INTEGER) <= ?")
                    params.append(grad_year_max)
                where_clause = " AND ".join(conditions)
                sql = f"SELECT DISTINCT candidate_id FROM education_history WHERE end_date IS NOT NULL AND {where_clause}"
                rows = conn.execute(sql, params).fetchall()
                grad_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 毕业年份过滤: year∈[{grad_year_min or '*'}, {grad_year_max or '*'}] → {len(grad_ids)}人")
                candidate_ids = grad_ids if candidate_ids is None else candidate_ids & grad_ids

        # 全日制约束（基于 education_history.is_fulltime）
        if constraints.get("require_fulltime"):
            with hr_db._get_conn() as conn:
                # 排除最高学历为非全日制的候选人
                # 逻辑：每个候选人的最后一条 education_history（最高学历）必须是全日制
                sql = """SELECT DISTINCT candidate_id FROM education_history 
                         WHERE is_fulltime = 1 
                         AND candidate_id NOT IN (
                             SELECT e1.candidate_id FROM education_history e1
                             WHERE e1.end_date = (
                                 SELECT MAX(e2.end_date) FROM education_history e2 
                                 WHERE e2.candidate_id = e1.candidate_id
                             ) AND e1.is_fulltime = 0
                         )"""
                rows = conn.execute(sql).fetchall()
                fulltime_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 全日制过滤: {len(fulltime_ids)}人满足")
                candidate_ids = fulltime_ids if candidate_ids is None else candidate_ids & fulltime_ids

        # 学历约束（兼容 education_level 和 highest_education 两种列名）
        if "highest_education" in constraints:
            edu = constraints["highest_education"]
            # 学历有层级关系：博士 > 硕士 > 本科
            edu_hierarchy = {"本科": ["本科", "硕士", "博士"], "硕士": ["硕士", "博士"], "博士": ["博士"]}
            valid_edus = edu_hierarchy.get(edu, [edu])
            placeholders = ",".join(["?" for _ in valid_edus])
            with hr_db._get_conn() as conn:
                # 尝试两种列名
                try:
                    rows = conn.execute(
                        f"SELECT id FROM candidates WHERE highest_education IN ({placeholders})",
                        valid_edus
                    ).fetchall()
                except Exception:
                    rows = conn.execute(
                        f"SELECT id FROM candidates WHERE education_level IN ({placeholders})",
                        valid_edus
                    ).fetchall()
                edu_ids = {r[0] for r in rows}
                candidate_ids = edu_ids if candidate_ids is None else candidate_ids & edu_ids

        # 论文发表等级约束（CCF-A/B/C）
        if "min_pub_rank" in constraints:
            rank = constraints["min_pub_rank"]
            # CCF-A 只匹配 A；CCF-B 匹配 A+B；CCF-C 匹配 A+B+C
            rank_hierarchy = {
                "CCF-A": ["CCF-A"],
                "CCF-B": ["CCF-A", "CCF-B"],
                "CCF-C": ["CCF-A", "CCF-B", "CCF-C"],
            }
            valid_ranks = rank_hierarchy.get(rank, [rank])
            placeholders = ",".join(["?" for _ in valid_ranks])
            with hr_db._get_conn() as conn:
                sql = f"SELECT DISTINCT candidate_id FROM publications WHERE venue_rank IN ({placeholders})"
                rows = conn.execute(sql, valid_ranks).fetchall()
                pub_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 论文等级过滤({rank}): {len(pub_ids)}人满足")
                candidate_ids = pub_ids if candidate_ids is None else candidate_ids & pub_ids

        # SCI分区约束
        if "max_sci_zone" in constraints:
            zone = constraints["max_sci_zone"]
            # Q1 只匹配 Q1；Q2 匹配 Q1+Q2；以此类推
            zone_hierarchy = {
                "Q1": ["Q1"],
                "Q2": ["Q1", "Q2"],
                "Q3": ["Q1", "Q2", "Q3"],
                "Q4": ["Q1", "Q2", "Q3", "Q4"],
            }
            valid_zones = zone_hierarchy.get(zone, [zone])
            placeholders = ",".join(["?" for _ in valid_zones])
            with hr_db._get_conn() as conn:
                sql = f"SELECT DISTINCT candidate_id FROM publications WHERE sci_zone IN ({placeholders})"
                rows = conn.execute(sql, valid_zones).fetchall()
                sci_ids = {r[0] for r in rows}
                logger.info(f"[RAG] SCI分区过滤({zone}): {len(sci_ids)}人满足")
                candidate_ids = sci_ids if candidate_ids is None else candidate_ids & sci_ids

        # 通用论文约束（仅要求有论文）
        if constraints.get("has_publications"):
            with hr_db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT candidate_id FROM publications"
                ).fetchall()
                pub_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 有论文过滤: {len(pub_ids)}人有论文发表")
                candidate_ids = pub_ids if candidate_ids is None else candidate_ids & pub_ids

        # 国际会议等级约束
        if "min_conf_rank" in constraints:
            rank = constraints["min_conf_rank"]
            rank_hierarchy = {
                "CCF-A": ["CCF-A"],
                "CCF-B": ["CCF-A", "CCF-B"],
                "CCF-C": ["CCF-A", "CCF-B", "CCF-C"],
            }
            valid_ranks = rank_hierarchy.get(rank, [rank])
            placeholders = ",".join(["?" for _ in valid_ranks])
            with hr_db._get_conn() as conn:
                sql = f"SELECT DISTINCT candidate_id FROM conferences WHERE conference_rank IN ({placeholders})"
                rows = conn.execute(sql, valid_ranks).fetchall()
                conf_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 会议等级过滤({rank}): {len(conf_ids)}人满足")
                candidate_ids = conf_ids if candidate_ids is None else candidate_ids & conf_ids

        # 通用国际会议约束
        if constraints.get("has_conferences"):
            with hr_db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT candidate_id FROM conferences"
                ).fetchall()
                conf_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 有会议经历过滤: {len(conf_ids)}人有国际会议记录")
                candidate_ids = conf_ids if candidate_ids is None else candidate_ids & conf_ids

        # 作者位次约束
        if "author_position" in constraints:
            pos = constraints["author_position"]
            with hr_db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT candidate_id FROM publications WHERE author_position = ?",
                    (pos,)
                ).fetchall()
                pos_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 作者位次过滤({pos}): {len(pos_ids)}人满足")
                candidate_ids = pos_ids if candidate_ids is None else candidate_ids & pos_ids

        # ══════════════════════════════════════════════════════════════════════
        # 动态扩展属性过滤（LLM 提取的 extra_constraints）
        # 通用引擎：根据 operator 对 candidate_extra_attributes 表做过滤
        # ══════════════════════════════════════════════════════════════════════
        extra_constraints = constraints.get("_extra_constraints", {})
        if extra_constraints:
            extra_matched_ids = self._filter_by_extra_attributes(extra_constraints)
            if extra_matched_ids is not None:
                logger.info(f"[RAG] 动态属性过滤: {len(extra_matched_ids)}人满足 {list(extra_constraints.keys())}")
                candidate_ids = extra_matched_ids if candidate_ids is None else candidate_ids & extra_matched_ids

        # 返回值语义：
        # - None: 没有任何硬约束触发SQL过滤（应跳过硬约束逻辑）
        # - 非空 set: 有候选人满足所有硬约束
        # - 空 set(): 有硬约束但没有候选人满足交集
        return candidate_ids

    def _sql_prefilter_soft(self, soft_constraints: Dict[str, Any]) -> Optional[Set[int]]:
        """基于软约束查询匹配的候选人 ID 集合（用于评分加权，不做过滤）。
        
        与 _sql_prefilter 的区别：
        - 多个软约束之间用 OR（并集）而非 AND（交集）
        - 结果仅用于 _fuse_results 中的评分加分，不用于过滤
        """
        all_soft_ids = set()
        
        # 院校层级软约束
        if "school_tier" in soft_constraints:
            tier = soft_constraints["school_tier"]
            with hr_db._get_conn() as conn:
                if tier == "985":
                    tier_condition = "school_tier = '985'"
                elif tier == "211":
                    tier_condition = "school_tier IN ('985', '211')"
                else:
                    tier_condition = "school_tier IN ('985', '211', '双一流')"
                rows = conn.execute(
                    f"SELECT DISTINCT candidate_id FROM education_history WHERE {tier_condition}"
                ).fetchall()
                tier_ids = {r[0] for r in rows}
                logger.info(f"[RAG] 软约束-院校层级({tier}): {len(tier_ids)}人匹配")
                all_soft_ids |= tier_ids
        
        # 学校软约束
        if "school" in soft_constraints:
            school = soft_constraints["school"]
            with hr_db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT candidate_id FROM education_history WHERE school LIKE ?",
                    (f"%{school}%",)
                ).fetchall()
                all_soft_ids |= {r[0] for r in rows}
        
        # 学历软约束
        if "highest_education" in soft_constraints:
            edu = soft_constraints["highest_education"]
            edu_hierarchy = {"本科": ["本科", "硕士", "博士"], "硕士": ["硕士", "博士"], "博士": ["博士"]}
            valid_edus = edu_hierarchy.get(edu, [edu])
            placeholders = ",".join(["?" for _ in valid_edus])
            with hr_db._get_conn() as conn:
                try:
                    rows = conn.execute(
                        f"SELECT id FROM candidates WHERE highest_education IN ({placeholders})",
                        valid_edus
                    ).fetchall()
                except Exception:
                    rows = conn.execute(
                        f"SELECT id FROM candidates WHERE education_level IN ({placeholders})",
                        valid_edus
                    ).fetchall()
                all_soft_ids |= {r[0] for r in rows}
        
        # 工作年限软约束
        if "min_work_years" in soft_constraints:
            min_years = soft_constraints["min_work_years"]
            with hr_db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM candidates WHERE work_years >= ?", (min_years,)
                ).fetchall()
                all_soft_ids |= {r[0] for r in rows}
        
        # 论文软约束
        if soft_constraints.get("has_publications"):
            with hr_db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT candidate_id FROM publications"
                ).fetchall()
                all_soft_ids |= {r[0] for r in rows}
        
        return all_soft_ids if all_soft_ids else None

    def _filter_by_extra_attributes(self, extra_constraints: Dict[str, Any]) -> Optional[Set[int]]:
        """通用动态属性过滤引擎
        
        根据 LLM 提取的 extra_constraints 对 candidate_extra_attributes 表进行过滤。
        支持任意 attr_key 和多种比较操作符。
        
        参数:
            extra_constraints: {
                "gpa": {"operator": ">=", "value": 3.0},
                "target_job": {"operator": "contains", "value": "DevOps"},
                ...
            }
            
        返回:
            满足所有动态约束的候选人ID集合，或 None（无有效约束时）
        """
        if not extra_constraints:
            return None
            
        result_ids = None
        
        for attr_key, condition in extra_constraints.items():
            if not isinstance(condition, dict):
                continue
                
            operator = condition.get("operator", "==")
            value = condition.get("value")
            if value is None:
                continue
            
            with hr_db._get_conn() as conn:
                # 防御：若该属性维度在整张表中根本不存在（数据集无法表达此条件），
                # 应跳过该约束，而不是因 0 命中导致整体交集归零。
                # 典型场景：LLM 把"海外留学经历"幻觉成 overseas_experience 等
                # 数据库并不存在的 attr_key——此时交由 has_overseas_edu 等
                # 标准约束路径处理，避免误把全部候选人过滤为空。
                attr_exists = conn.execute(
                    "SELECT 1 FROM candidate_extra_attributes WHERE attr_key = ? LIMIT 1",
                    (attr_key,)
                ).fetchone()
                if attr_exists is None:
                    logger.warning(
                        f"[RAG] 动态属性 '{attr_key}' 在数据集中不存在，跳过该约束"
                        f"（不参与交集，避免误清空结果）"
                    )
                    continue

                # 先获取所有拥有该属性的候选人
                rows = conn.execute(
                    "SELECT candidate_id, attr_value, attr_type FROM candidate_extra_attributes WHERE attr_key = ?",
                    (attr_key,)
                ).fetchall()
                
                matched = set()
                for row in rows:
                    cid = row["candidate_id"]
                    stored_value = row["attr_value"]
                    stored_type = row["attr_type"]
                    
                    if self._compare_attr_value(stored_value, stored_type, operator, value):
                        matched.add(cid)
                
                logger.info(f"[RAG] 动态属性 '{attr_key}' {operator} {value}: "
                           f"共{len(rows)}人有此属性, {len(matched)}人满足条件")
                
                if result_ids is None:
                    result_ids = matched
                else:
                    result_ids = result_ids & matched
        
        return result_ids

    def _compare_attr_value(self, stored_value: str, stored_type: str,
                            operator: str, target_value: Any) -> bool:
        """比较存储的属性值与目标值
        
        支持数值比较（>=, <=, >, <, ==）和字符串包含匹配（contains）。
        对于数值比较，会尝试将存储值转为浮点数。
        """
        try:
            if operator == "contains":
                # 字符串包含匹配（不区分大小写）
                return str(target_value).lower() in str(stored_value).lower()
            
            elif operator == "==":
                # 精确匹配
                if stored_type == "number":
                    return float(stored_value) == float(target_value)
                return str(stored_value).strip().lower() == str(target_value).strip().lower()
            
            elif operator in (">=", "<=", ">", "<"):
                # 数值比较
                stored_num = float(stored_value)
                target_num = float(target_value)
                if operator == ">=":
                    return stored_num >= target_num
                elif operator == "<=":
                    return stored_num <= target_num
                elif operator == ">":
                    return stored_num > target_num
                elif operator == "<":
                    return stored_num < target_num
            
            return False
        except (ValueError, TypeError):
            # 类型转换失败时，尝试字符串比较
            if operator == "contains":
                return str(target_value).lower() in str(stored_value).lower()
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # BM25 检索
    # ══════════════════════════════════════════════════════════════════════════

    # 候选人文本缓存 (candidate_id -> tokenized_terms)
    _text_cache: Dict[int, List[str]] = {}
    _data_cache: Dict[int, Dict[str, Any]] = {}

    def _ensure_cache_loaded(self):
        """一次性加载所有候选人的文本索引到内存缓存"""
        if self._text_cache:
            return  # 已加载

        logger.info("[RAG] 首次加载候选人文本缓存...")
        # 必须加载全部候选人——数据量已超过 2000，写死 limit=1000 会导致
        # 约一半候选人永远无法被 BM25 检索/约束注入命中。
        candidates = hr_db.search_candidates(limit=1000000)
        for cand in candidates:
            cid = cand['id']
            cand_full = hr_db.get_candidate(cid)
            if not cand_full:
                continue

            self._data_cache[cid] = cand_full

            # 构建完整的候选人文本索引
            text_parts = [
                cand_full.get('name', ''),
                cand_full.get('education_level', '') or cand_full.get('highest_education', ''),
                cand_full.get('school', '') or '',
                cand_full.get('major', '') or '',
            ]
            # 加入完整教育经历
            for edu in cand_full.get('education_history', []):
                text_parts.append(edu.get('school', ''))
                text_parts.append(edu.get('major', ''))
                text_parts.append(edu.get('degree', ''))
                if edu.get('school_tier'):
                    text_parts.append(edu.get('school_tier', ''))
            # 加入技能
            for s in cand_full.get('skills', []):
                text_parts.append(s.get('skill_name', ''))
            # 加入工作经历
            for exp in cand_full.get('work_experiences', []):
                text_parts.append(exp.get('position', ''))
                text_parts.append(exp.get('company_name', ''))
                text_parts.append(exp.get('description', ''))
            # 加入获奖证书
            for award in cand_full.get('awards_certificates', []):
                text_parts.append(award.get('name', '') or award.get('award_name', ''))
            # 加入项目
            for proj in cand_full.get('projects', []):
                text_parts.append(proj.get('project_name', ''))
                techs = proj.get('technologies', '')
                if techs:
                    text_parts.append(techs if isinstance(techs, str) else ','.join(techs))
                text_parts.append(proj.get('description', ''))
            # 加入论文发表
            for pub in cand_full.get('publications', []):
                text_parts.append(pub.get('title', ''))
                text_parts.append(pub.get('venue', ''))
                text_parts.append(pub.get('venue_rank', ''))
                if pub.get('sci_zone'):
                    text_parts.append(f"SCI {pub['sci_zone']}")
                text_parts.append(pub.get('author_position', ''))
            # 加入国际会议
            for conf in cand_full.get('conferences', []):
                text_parts.append(conf.get('conference_name', ''))
                text_parts.append(conf.get('conference_rank', ''))
                text_parts.append(conf.get('role', ''))
                if conf.get('paper_title'):
                    text_parts.append(conf['paper_title'])

            # 过滤 None/非字符串项（加载全量数据后部分字段可能为 NULL）
            text = ' '.join(str(p) for p in text_parts if p)
            self._text_cache[cid] = self._tokenize(text)

        logger.info(f"[RAG] 缓存加载完成: {len(self._text_cache)} 个候选人")

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25稀疏检索 — 基于缓存的候选人完整文本"""
        self._ensure_cache_loaded()
        query_terms = self._expand_synonyms(self._tokenize(query))

        N = len(self._text_cache)
        scored = []
        for cid, doc_terms in self._text_cache.items():
            score = self._compute_bm25(query_terms, doc_terms, N)
            scored.append({"candidate_id": cid, "score": score, "data": self._data_cache.get(cid)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ══════════════════════════════════════════════════════════════════════════
    # Dense 向量检索
    # ══════════════════════════════════════════════════════════════════════════

    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BGE-M3稠密向量检索"""
        query_embedding = multimodal_fusion.extract_text_features(query).flatten().tolist()
        results = vector_db.search_similar(query_embedding, top_k)
        dense_results = []
        for r in results:
            cand = hr_db.get_candidate(r["candidate_id"])
            if cand:
                dense_results.append({
                    "candidate_id": r["candidate_id"],
                    "score": 1.0 - r.get("distance", 0.5),
                    "data": cand
                })
        return dense_results

    # ══════════════════════════════════════════════════════════════════════════
    # 融合排序
    # ══════════════════════════════════════════════════════════════════════════

    def _fuse_results(self, bm25: List, dense: List, top_k: int,
                      constraint_ids: Optional[Set[int]] = None,
                      soft_constraint_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """加权融合两路结果 + 硬约束优先策略 + 软约束加分
        
        当检测到硬约束时：
        - 将满足硬约束的候选人优先排序在前
        - 在硬约束候选人内部按 BM25+Dense 融合分排序
        - 硬约束候选人后面再接非硬约束候选人
        这确保了"精确匹配的人一定排在模糊匹配的人前面"
        
        当检测到软约束时：
        - 满足软约束的候选人获得额外加分（SOFT_BOOST=0.15）
        - 不做过滤，仅影响排序
        """
        # 对两路分数分别做 min-max 归一化到 [0, 1]
        bm25_scores = [item["score"] for item in bm25] if bm25 else [0]
        dense_scores = [item["score"] for item in dense] if dense else [0]

        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        dense_min, dense_max = min(dense_scores), max(dense_scores)

        def normalize_bm25(score):
            if bm25_max == bm25_min:
                return 1.0 if score > 0 else 0.0
            return (score - bm25_min) / (bm25_max - bm25_min)

        def normalize_dense(score):
            if dense_max == dense_min:
                return 1.0 if score > 0 else 0.0
            return (score - dense_min) / (dense_max - dense_min)

        score_map = {}
        for item in bm25:
            cid = item["candidate_id"]
            score_map[cid] = {
                "bm25": normalize_bm25(item["score"]),
                "dense": 0,
                "data": item.get("data")
            }
        for item in dense:
            cid = item["candidate_id"]
            if cid in score_map:
                score_map[cid]["dense"] = normalize_dense(item["score"])
            else:
                score_map[cid] = {
                    "bm25": 0,
                    "dense": normalize_dense(item["score"]),
                    "data": item.get("data")
                }

        # 分两组：满足硬约束 vs 不满足
        matched_items = []
        unmatched_items = []
        
        SOFT_BOOST = 0.15  # 软约束加分系数

        for cid, scores in score_map.items():
            base_score = RAG_BM25_WEIGHT * scores["bm25"] + RAG_DENSE_WEIGHT * scores["dense"]
            is_match = bool(constraint_ids and cid in constraint_ids)
            
            # 软约束加分：满足偏好条件的候选人获得额外分数
            soft_match = bool(soft_constraint_ids and cid in soft_constraint_ids)
            if soft_match:
                base_score += SOFT_BOOST
            
            item = {
                "candidate_id": cid,
                "score": base_score,
                "data": scores["data"],
                "constraint_match": is_match,
                "soft_match": soft_match,
            }
            
            if is_match:
                matched_items.append(item)
            else:
                unmatched_items.append(item)

        # 两组各自按分数排序
        matched_items.sort(key=lambda x: x["score"], reverse=True)
        unmatched_items.sort(key=lambda x: x["score"], reverse=True)

        # 策略：硬约束是 must 条件，满足硬约束的候选人必须始终优先于不满足者。
        # 无论匹配数多少，都将满足硬约束者整体排在不满足者之前；约束组内部、
        # 非约束组内部各自按融合分排序。这样既能在约束命中很多人时正确返回
        # 约束内最相关的 top_k，也能在约束命中很少时把全部约束候选人顶到前面。
        if constraint_ids:
            if unmatched_items:
                max_unmatched_score = unmatched_items[0]["score"]
            else:
                max_unmatched_score = 0
            # 给硬约束候选人一个确保排在非约束候选人之上的分数偏移
            boost = max(max_unmatched_score + 0.01, 0.5)
            for item in matched_items:
                item["score"] = item["score"] + boost
            fused = matched_items + unmatched_items
        else:
            fused = matched_items + unmatched_items
            fused.sort(key=lambda x: x["score"], reverse=True)

        return fused[:top_k]

    # ══════════════════════════════════════════════════════════════════════════
    # 同义词表 & 工具方法
    # ══════════════════════════════════════════════════════════════════════════

    # 高校简称→全称映射表（用于 BM25 查询扩展）
    _SCHOOL_SYNONYMS = {
        "川大": "四川大学", "北大": "北京大学", "清华": "清华大学",
        "复旦": "复旦大学", "上交": "上海交通大学", "交大": "上海交通大学",
        "浙大": "浙江大学", "南大": "南京大学", "中科大": "中国科学技术大学",
        "哈工大": "哈尔滨工业大学", "西交": "西安交通大学",
        "武大": "武汉大学", "华科": "华中科技大学", "中山": "中山大学",
        "同济": "同济大学", "北航": "北京航空航天大学", "北理": "北京理工大学",
        "华南理工": "华南理工大学", "电子科大": "电子科技大学",
        "成电": "电子科技大学", "西工大": "西北工业大学",
        "国防科大": "国防科技大学", "东南": "东南大学",
        "厦大": "厦门大学", "天大": "天津大学", "南开": "南开大学",
        "吉大": "吉林大学", "山大": "山东大学", "兰大": "兰州大学",
        "川师": "四川师范大学", "成理": "成都理工大学",
        "西南交大": "西南交通大学", "重邮": "重庆邮电大学",
        "国科大": "中国科学院大学", "山西大": "山西大学",
        "北邮": "北京邮电大学", "杭电": "杭州电子科技大学",
        "深大": "深圳大学", "南邮": "南京邮电大学",
        "西电": "西安电子科技大学", "中南": "中南大学",
        "湖大": "湖南大学", "重大": "重庆大学",
    }

    # 技术方向同义词（用于查询扩展）
    _DIRECTION_SYNONYMS = {
        "后端": ["后端", "backend", "服务端", "服务器", "Java", "Spring", "Python", "Go", "微服务", "分布式"],
        "前端": ["前端", "frontend", "React", "Vue", "JavaScript", "TypeScript", "HTML", "CSS"],
        "算法": ["算法", "机器学习", "深度学习", "AI", "人工智能", "NLP", "CV", "推荐系统"],
        "运维": ["运维", "DevOps", "SRE", "Kubernetes", "Docker", "CI/CD", "Linux"],
        "数据": ["数据", "大数据", "Spark", "Hadoop", "Flink", "数据仓库", "ETL"],
        "测试": ["测试", "QA", "自动化测试", "性能测试", "单元测试"],
        "移动端": ["iOS", "Android", "移动端", "Flutter", "React Native"],
    }

    def _expand_synonyms(self, terms: List[str]) -> List[str]:
        """对查询 term 做同义词扩展（简称→全称 + 方向扩展）"""
        expanded = list(terms)

        for term in terms:
            # 院校简称扩展
            full_name = self._SCHOOL_SYNONYMS.get(term)
            if full_name:
                expanded.extend(
                    w for w in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', full_name.lower()) if len(w) > 1
                )

            # 技术方向扩展（"后端" → 添加相关技术词）
            direction_terms = self._DIRECTION_SYNONYMS.get(term)
            if direction_terms:
                for dt in direction_terms:
                    expanded.extend(
                        w for w in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', dt.lower()) if len(w) > 1
                    )

        return list(set(expanded))  # 去重

    def _tokenize(self, text: str) -> List[str]:
        """中文分词（jieba）+ 英文单词提取

        使用 jieba 对中文进行词级分割，同时保留英文单词。
        过滤单字和停用词，保留 2 字以上的有意义 token。
        """
        # jieba 分词（精确模式）
        tokens = list(jieba.cut(text))
        # 过滤：保留 2 字以上的中文词和英文单词，去除标点/数字/单字
        result = []
        for w in tokens:
            w = w.strip().lower()
            if len(w) < 2:
                continue
            # 中文词（至少 2 字）
            if re.match(r'^[\u4e00-\u9fff]{2,}$', w):
                result.append(w)
            # 英文单词（至少 2 字母，允许数字如 c++）
            elif re.match(r'^[a-z][a-z0-9+#.]*$', w) and len(w) >= 2:
                result.append(w)
        return result

    def _compute_bm25(self, query_terms: List[str], doc_terms: List[str], N: int,
                      k1: float = 1.5, b: float = 0.75) -> float:
        """计算BM25分数"""
        if not doc_terms:
            return 0.0
        tf = Counter(doc_terms)
        dl = len(doc_terms)
        avgdl = 50  # 近似平均文档长度
        score = 0.0
        for term in query_terms:
            if term in tf:
                idf = math.log((N - 1 + 0.5) / (1 + 0.5))
                term_freq = tf[term]
                numerator = term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * dl / avgdl)
                score += idf * numerator / denominator
        return score
