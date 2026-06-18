"""匹配评估Skill - 硬性规则过滤+多模态分层融合匹配 (核心创新点2实现)"""
import logging
import re
import numpy as np
from typing import Dict, Any, List
from backend.skills.base_skill import BaseSkill
from backend.models.multimodal_fusion import multimodal_fusion
from backend.models.catboost_matcher import catboost_matcher
from backend.database.models import hr_db
from backend.config import FINAL_TOP_K
from backend.utils.candidate_category import (
    get_grad_year_filter_for_query,
    get_fresh_grad_year_range,
    get_intern_grad_year_min,
)
from backend.utils.school_tier import (
    school_tier_classifier,
    get_tier_rank,
    TIER_985,
    TIER_211,
    TIER_SHUANG,
)

logger = logging.getLogger(__name__)


class MatchingEvaluationSkill(BaseSkill):
    """匹配评估Skill: 硬性过滤 -> 多模态分层融合 -> Top10"""

    def __init__(self):
        super().__init__(name="matching_evaluation", description="多模态分层融合人岗匹配")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        candidates = params.get("candidates", {}).get("candidates", [])
        jd_info = params.get("jd_info", {})
        context = params.get("context", {})

        if not candidates:
            # 如果没有预检索的候选人,从数据库获取
            from backend.skills.rag_retrieval_skill import RAGRetrievalSkill
            rag = RAGRetrievalSkill()
            rag_result = await rag.execute({"query": query})
            candidates = rag_result.get("candidates", [])

        # 从查询中自动解析 jd_info（如用户未显式提供）
        if not jd_info.get("min_experience"):
            jd_info = self._parse_jd_from_query(query, jd_info)

        # Step 1: 硬性规则过滤
        filtered = self._apply_hard_rules(candidates, jd_info)
        logger.info(f"[Matching] After hard rules: {len(filtered)}/{len(candidates)}")

        # Step 2: 多模态分层融合匹配评分
        scored = self._multimodal_matching(query, filtered, jd_info)

        # Step 3: 排序取Top10
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        top_results = scored[:FINAL_TOP_K]

        return {
            "matched_candidates": top_results,
            "total_evaluated": len(filtered),
            "top_k": len(top_results),
            "method": "multimodal_hierarchical_fusion"
        }

    # 常见院校别名映射表
    SCHOOL_ALIASES = {
        "上交": "上海交通大学", "交大": "上海交通大学", "上海交大": "上海交通大学",
        "电子科大": "电子科技大学", "成电": "电子科技大学",
        "川大": "四川大学",
        "北邮": "北京邮电大学",
        "哈工大": "哈尔滨工业大学",
        "华科": "华中科技大学",
        "浙大": "浙江大学",
        "北大": "北京大学",
        "清华": "清华大学",
        "复旦": "复旦大学",
        "南大": "南京大学",
        "中科大": "中国科学技术大学",
        "西电": "西安电子科技大学",
        "武大": "武汉大学",
        "同济": "同济大学",
        "南开": "南开大学",
        "中山": "中山大学", "中大": "中山大学",
        "厦大": "厦门大学",
        "人大": "中国人民大学",
        "北航": "北京航空航天大学",
        "北理工": "北京理工大学",
        "哈工程": "哈尔滨工程大学",
        "西工大": "西北工业大学",
    }

    def _parse_jd_from_query(self, query: str, jd_info: Dict) -> Dict:
        """从自然语言查询中解析 JD 信息（年限、技能、学历、目标院校等）
        
        增强版：支持从完整JD长文本中提取结构化信息，
        包括岗位类型、核心技能、实习生/应届生标识等。
        """
        jd = dict(jd_info)  # 复制避免修改原始

        # ── 实习生/应届生场景检测（动态时间，不硬编码年份） ─────────────────
        category_filters = get_grad_year_filter_for_query(query)
        if category_filters.get("is_intern"):
            jd["is_intern"] = True
            jd["max_experience"] = category_filters.get("max_work_years", 1)
        elif category_filters.get("is_fresh_grad"):
            jd["is_fresh_grad"] = True
            jd["max_experience"] = category_filters.get("max_work_years", 2)
        if "grad_year_min" in category_filters:
            jd["grad_year_min"] = category_filters["grad_year_min"]
        if "grad_year_max" in category_filters:
            jd["grad_year_max"] = category_filters["grad_year_max"]

        # ── 解析经验年限 ──────────────────────────────────────────────────────
        # 注意：排除看起来像年份的数字（如2010年、2015-2026年）
        cn_num_map = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                      "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        has_year_range = re.search(r"\d{4}\s*[-–~到至]+\s*\d{4}\s*年", query)
        has_graduation_context = re.search(r"\d{4}\s*年?\s*(?:之?[后以]|毕业|之间|间)", query)

        is_intern = category_filters.get("is_intern", False)
        if not is_intern:  # 实习生不解析min_experience
            exp_match = re.search(r"(?<!\d)(\d{1,2})\s*年(?:以上|经验|工作)", query)
            if not exp_match and not has_year_range and not has_graduation_context:
                exp_match = re.search(r"(?<!\d)(\d{1,2})\s*年", query)
                if exp_match and int(exp_match.group(1)) > 30:
                    exp_match = None
            if exp_match:
                jd["min_experience"] = int(exp_match.group(1))
            else:
                cn_match = re.search(r"([一二两三四五六七八九十]+)\s*年(?:以上|经验|工作)?", query)
                if cn_match:
                    cn_num = cn_match.group(1)
                    jd["min_experience"] = cn_num_map.get(cn_num, 0)

        # ── 解析技术栈关键词（增强版：覆盖数据/AI领域） ──────────────────────
        tech_pattern = (
            r"(Java|Python|Go|Golang|C\+\+|C/C\+\+|JavaScript|TypeScript|React|Vue|"
            r"Spring|MySQL|Redis|Kafka|Docker|K8s|Kubernetes|Node\.?js|"
            r"PyTorch|TensorFlow|Sklearn|scikit-learn|Spark|Hive|Hadoop|Flink|"
            r"Pandas|NumPy|SQL|R语言|CUDA|Transformer|BERT|GPT|"
            r"Elasticsearch|MongoDB|PostgreSQL|Linux|Git)"
        )
        tech_matches = re.findall(tech_pattern, query, re.IGNORECASE)
        if tech_matches:
            # 去重
            existing_skills = set(s.lower() for s in jd.get("required_skills", []))
            new_skills = [s for s in tech_matches if s.lower() not in existing_skills]
            jd["required_skills"] = jd.get("required_skills", []) + new_skills

        # ── 解析职位类型关键词（增强版：更精准的数据/AI岗识别） ────────────────
        # 优先匹配更具体的复合关键词
        position_type_patterns = [
            (r"数据挖掘", "数据"),
            (r"机器学习|深度学习|强化学习", "算法"),
            (r"自然语言处理|NLP|大模型|LLM", "算法"),
            (r"计算机视觉|CV|图像识别", "算法"),
            (r"数据分析|数据科学|数据开发|数据工程|BI", "数据"),
            (r"推荐系统|搜索算法|广告算法", "算法"),
            (r"前端|Frontend", "前端"),
            (r"后端|Backend|服务端", "后端"),
            (r"全栈|Full.?Stack", "全栈"),
            (r"AI|人工智能", "AI"),
            (r"移动端|客户端", "移动端"),
            (r"iOS", "iOS"),
            (r"Android", "Android"),
            (r"测试|QA", "测试"),
            (r"运维|DevOps|SRE", "运维"),
            (r"架构", "架构"),
            (r"产品", "产品"),
        ]
        for pattern, ptype in position_type_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                jd["position_type"] = ptype
                break

        # 解析目标院校（包括别名）
        target_schools = []
        for alias, full_name in self.SCHOOL_ALIASES.items():
            if alias in query:
                if full_name not in target_schools:
                    target_schools.append(full_name)
        # 直接匹配完整校名
        known_schools = list(set(self.SCHOOL_ALIASES.values()))
        for school in known_schools:
            if school in query and school not in target_schools:
                target_schools.append(school)
        if target_schools:
            jd["target_schools"] = target_schools

        # 解析院校层级要求（985/211/双一流）
        tier_patterns = [
            (r'[Cc]9|C9联盟|九校联盟', '985'),
            (r'985', '985'),
            (r'211', '211'),
            (r'双一流|"双一流"', '双一流'),
        ]
        for pattern, tier in tier_patterns:
            if re.search(pattern, query):
                jd["required_school_tier"] = tier
                break

        # 解析毕业年份范围
        grad_year_match = re.search(r"(\d{4})[-–~到至]+(\d{4})", query)
        if grad_year_match:
            jd["grad_year_min"] = int(grad_year_match.group(1))
            jd["grad_year_max"] = int(grad_year_match.group(2))
        else:
            # "2010年之后毕业" 模式
            after_match = re.search(r"(\d{4})\s*年?[之以]?后", query)
            if after_match:
                jd["grad_year_min"] = int(after_match.group(1))

        # ── 解析全日制/非全日制偏好 ─────────────────────────────────────────────
        # "不要非全日制"、"排除非全"、"非非全日制"、"全日制优先"、"必须全日制"
        if re.search(r"不要?非全日?制?|排除非全|非全日?制?除外|必须全日制|只要全日制|要求全日制|全日制[优優]先", query):
            jd["require_fulltime"] = True
        elif re.search(r"非全日?制?[也亦]可|接受非全|非全日?制?优先", query):
            jd["accept_parttime"] = True
        elif re.search(r"全日制", query):
            # "全日制硕士"、"全日制本科" 等正面要求
            jd["require_fulltime"] = True

        logger.info(f"[Matching] Parsed JD from query: {jd}")
        return jd

    def _apply_hard_rules(self, candidates: List[Dict], jd_info: Dict) -> List[Dict]:
        """应用硬性规则过滤（院校匹配加权、年限不足降权但不完全排除，确保有结果返回）
        
        增强：支持实习生/应届生场景的max_experience过滤 + 毕业年份硬过滤
        """
        hard_rules = jd_info.get("hard_rules", [])

        # 即使没有显式 hard_rules，也根据解析出的 min_experience 进行软过滤
        min_exp = jd_info.get("min_experience", 0)
        max_exp = jd_info.get("max_experience", 9999)  # 实习生/应届生场景：最大经验限制
        is_intern = jd_info.get("is_intern", False)
        is_fresh_grad = jd_info.get("is_fresh_grad", False)
        target_schools = jd_info.get("target_schools", [])
        grad_year_min = jd_info.get("grad_year_min", 0)
        grad_year_max = jd_info.get("grad_year_max", 9999)
        require_fulltime = jd_info.get("require_fulltime", False)
        required_school_tier = jd_info.get("required_school_tier", "")

        filtered = []
        for cand in candidates:
            data = cand.get("data", cand)
            passed = True

            # 显式硬性规则
            for rule in hard_rules:
                field = rule.get("field", "")
                op = rule.get("operator", "=")
                value = rule.get("value", "")
                cand_value = str(data.get(field, ""))
                if op == "!=" and cand_value == value:
                    passed = False
                    break
                elif op == "not_contains" and value in cand_value:
                    passed = False
                    break
                elif op == ">=" and cand_value.isdigit() and int(cand_value) < int(value):
                    passed = False
                    break

            if passed:
                # 实习生/应届生场景：工作经验超过上限的候选人不通过
                cand_years = data.get("work_years", 0) or 0
                if (is_intern or is_fresh_grad) and cand_years > max_exp:
                    continue  # 跳过经验过多的候选人

                # 标记是否满足年限要求（用于后续排序惩罚）
                cand["_meets_experience"] = (cand_years >= min_exp) if min_exp > 0 else True

                # 检查院校匹配（遍历 education_history）
                cand["_school_match"] = False
                cand["_grad_year_match"] = False
                edu_history = data.get("education_history", [])
                if target_schools and edu_history:
                    for edu in edu_history:
                        edu_school = edu.get("school", "")
                        if any(ts in edu_school or edu_school in ts for ts in target_schools):
                            cand["_school_match"] = True
                            break
                elif target_schools and not edu_history:
                    # 回退到单字段 school 匹配
                    cand_school = data.get("school", "")
                    if any(ts in cand_school or cand_school in ts for ts in target_schools):
                        cand["_school_match"] = True

                # 检查毕业年份范围（应届生/实习生场景为硬过滤）
                if grad_year_min or grad_year_max < 9999:
                    grad_year_matched = False
                    if edu_history:
                        for edu in edu_history:
                            end_date = edu.get("end_date", "")
                            if end_date:
                                try:
                                    grad_year = int(end_date[:4])
                                    if grad_year_min <= grad_year <= grad_year_max:
                                        grad_year_matched = True
                                        break
                                except (ValueError, IndexError):
                                    pass
                    else:
                        # 回退到 graduation_year 字段
                        grad_year = data.get("graduation_year", 0) or 0
                        if grad_year_min <= grad_year <= grad_year_max:
                            grad_year_matched = True

                    cand["_grad_year_match"] = grad_year_matched
                    # 应届/实习场景下，毕业年份不匹配 → 硬过滤
                    if (is_intern or is_fresh_grad) and not grad_year_matched:
                        continue
                else:
                    cand["_grad_year_match"] = True  # 无年份限制时默认满足

                # 检查全日制要求（遍历最高学历的 is_fulltime 字段）
                cand["_fulltime_match"] = True  # 默认满足
                if require_fulltime and edu_history:
                    # 检查最高学历（最后一段教育经历）是否为全日制
                    highest_edu = edu_history[-1] if edu_history else {}
                    if highest_edu.get("is_fulltime") == 0:
                        cand["_fulltime_match"] = False
                        # 硬过滤：用户明确要求全日制时，排除非全日制
                        continue

                # 检查院校层级要求（985/211/双一流）— 硬过滤
                # 用权威分类器(school_tier.py)按学校名重新判定层级，覆盖 LLM 主观值，
                # 保证“中国科学院大学算作985”等规则一致生效。
                cand["_school_tier_match"] = True  # 默认满足
                # 候选人最优院校层级（权威判定）：缓存到 cand 供后续分级加权复用
                best_tier = school_tier_classifier.best_tier_of_candidate(edu_history)
                cand["_best_school_tier"] = best_tier
                cand["_best_school_rank"] = get_tier_rank(best_tier)
                if required_school_tier:
                    # 定义层级包含关系
                    if required_school_tier == "985":
                        valid_tiers = {TIER_985}
                    elif required_school_tier == "211":
                        valid_tiers = {TIER_985, TIER_211}
                    else:  # 双一流
                        valid_tiers = {TIER_985, TIER_211, TIER_SHUANG}

                    if edu_history:
                        tier_matched = best_tier in valid_tiers
                    else:
                        # 回退到 candidates 表中的 school 字段（无法判断tier，通过）
                        tier_matched = True

                    cand["_school_tier_match"] = tier_matched
                    if not tier_matched:
                        continue  # 硬过滤：不满足院校层级要求

                filtered.append(cand)

        # 如果指定了目标院校，将匹配院校的候选人排在前面
        if target_schools:
            matched = [c for c in filtered if c.get("_school_match")]
            unmatched = [c for c in filtered if not c.get("_school_match")]
            logger.info(f"[Matching] School filter: {len(matched)} matched target schools, {len(unmatched)} others")
            filtered = matched + unmatched

        return filtered

    def _multimodal_matching(self, query: str, candidates: List[Dict], jd_info: Dict) -> List[Dict]:
        """多模态分层融合匹配"""
        min_exp = jd_info.get("min_experience", 0)
        max_exp = jd_info.get("max_experience", 9999)
        is_intern = jd_info.get("is_intern", False)
        is_fresh_grad = jd_info.get("is_fresh_grad", False)
        results = []
        for cand in candidates:
            data = cand.get("data", cand)
            cand_id = data.get("id", cand.get("candidate_id", 0))

            # 获取完整候选人信息（优先使用已加载的data，避免重复查库）
            if data and data.get("skills") is not None:
                full_cand = data
            else:
                full_cand = hr_db.get_candidate(cand_id) if cand_id else data

            if not full_cand:
                continue

            # 提取结构化特征
            structured_features = catboost_matcher.extract_structured_features(jd_info, full_cand)

            # 构建候选人文本
            cand_text = self._build_candidate_text(full_cand)

            # 获取图片路径（从获奖证书中获取）
            images = [a["image_path"] for a in full_cand.get("awards_certificates", [])
                      if a.get("image_path")]

            # 多模态融合匹配
            fusion_result = multimodal_fusion.compute_matching_score(
                jd_text=query,
                candidate_text=cand_text,
                candidate_images=images if images else None,
                structured_features=structured_features
            )

            # CatBoost结构化预测
            catboost_score = catboost_matcher.predict(structured_features)

            # 综合得分
            final_score = 0.6 * fusion_result["score"] + 0.4 * catboost_score

            # 年限惩罚/奖励
            cand_years = full_cand.get("work_years", 0) or 0
            if is_intern or is_fresh_grad:
                # 实习生/应届生场景：经验少是优势
                if cand_years <= max_exp:
                    final_score += 0.08  # 经验合理范围内加分
                else:
                    # 经验越多，对实习/应届岗越不匹配
                    final_score *= (1.0 - 0.2 * min(cand_years, 5) / 5)
            elif min_exp > 0:
                if cand_years >= min_exp:
                    # 满足要求，给予年限匹配奖励（经验越接近越好）
                    exp_bonus = min(0.05, (cand_years - min_exp) * 0.005)
                    final_score += exp_bonus
                else:
                    # 不满足要求，按缺口比例降权
                    gap_ratio = (min_exp - cand_years) / min_exp
                    final_score *= (1.0 - 0.3 * gap_ratio)  # 最多降30%

            # 目标院校匹配加分（显著提升排名）
            if cand.get("_school_match"):
                final_score += 0.15  # 院校精确匹配给予大幅加分

            # 院校层级分级加权：985 > 211 > 双一流 > 其它（国科大计入985）。
            # rank: 985=3, 211=2, 双一流=1, 其它=0；每级 +0.04，最高 +0.12。
            # 用 _best_school_rank（硬过滤阶段已权威判定并缓存）；缺失则现场补算。
            best_rank = cand.get("_best_school_rank")
            if best_rank is None:
                _bt = school_tier_classifier.best_tier_of_candidate(
                    full_cand.get("education_history", [])
                )
                best_rank = get_tier_rank(_bt)
                cand["_best_school_tier"] = _bt
                cand["_best_school_rank"] = best_rank
            tier_bonus = 0.04 * best_rank  # 985→0.12, 211→0.08, 双一流→0.04, 其它→0
            final_score += tier_bonus

            # 毕业年份匹配标记
            meets_grad_year = cand.get("_grad_year_match", True)

            results.append({
                "candidate_id": cand_id,
                "candidate": full_cand,
                "match_score": round(final_score, 4),
                "fusion_score": fusion_result["score"],
                "catboost_score": round(catboost_score, 4),
                "text_similarity": fusion_result.get("text_similarity", 0),
                "multimodal_similarity": fusion_result.get("multimodal_similarity", 0),
                "structured_features": structured_features.tolist(),
                "meets_experience_req": cand_years >= min_exp if min_exp > 0 else True,
                "school_match": cand.get("_school_match", False),
                "school_tier": cand.get("_best_school_tier", ""),
                "school_tier_rank": cand.get("_best_school_rank", 0),
                "grad_year_match": meets_grad_year,
            })
        return results

    def _build_candidate_text(self, cand: Dict) -> str:
        """构建候选人结构化简历文本（与 init_database.build_candidate_text 逻辑一致）

        格式: 个人简历 + 教育经历时间线 + 技术栈 + 获奖证书 + 工作经历 + 项目经历
        """
        sections = []

        # 个人信息
        personal_parts = [cand.get("name", ""), f"{cand.get('work_years', 0)}年工作经验"]
        if cand.get("current_position"):
            personal_parts.append(cand["current_position"])
        if cand.get("location"):
            personal_parts.append(f"坐标{cand['location']}")
        sections.append(f"个人简历：{'，'.join(personal_parts)}。")

        # 教育经历时间线
        edu_history = cand.get("education_history", [])
        if edu_history:
            edu_lines = []
            for edu in edu_history:
                start = edu.get('start_date', '').replace('-', '.')
                end = edu.get('end_date', '').replace('-', '.')
                school = edu.get('school', '')
                major = edu.get('major', '')
                degree = edu.get('degree', '')
                tier = edu.get('school_tier', '')
                ft_tag = "全日制" if edu.get("is_fulltime", True) else "非全日制"
                tier_str = f"{tier}/" if tier else ""
                edu_lines.append(f"{start}-{end}，{school}，{major}，{degree}（{tier_str}{ft_tag}）")
            sections.append(f"教育经历：{'；'.join(edu_lines)}。")
        else:
            parts = [cand.get("highest_education", ""), cand.get("school", ""), cand.get("major", "")]
            sections.append(f"教育经历：{' '.join(p for p in parts if p)}。")

        # 技术栈
        skills = cand.get("skills", [])
        if skills:
            proficiency_map = {5: "精通", 4: "熟练", 3: "熟悉", 2: "掌握", 1: "了解"}
            sorted_skills = sorted(skills, key=lambda s: s.get('proficiency', 0), reverse=True)
            skill_strs = [f"{s['skill_name']}({proficiency_map.get(s.get('proficiency', 1), '了解')})"
                          for s in sorted_skills]
            sections.append(f"技术栈：{'、'.join(skill_strs)}。")

        # 获奖证书
        awards = cand.get("awards_certificates", [])
        if awards:
            award_strs = [f"{a.get('name', '')}({a.get('level', '')})" for a in awards]
            sections.append(f"获奖证书：{'、'.join(award_strs)}。")

        # 工作经历
        work_exps = cand.get("work_experiences", [])
        if work_exps:
            exp_lines = []
            for exp in work_exps:
                company = exp.get('company_name', '')
                pos = exp.get('position', '')
                duration = exp.get('duration_months', 0)
                desc = exp.get('description', '')
                duration_str = f"({duration}个月)" if duration else ""
                desc_str = f"-{desc}" if desc else ""
                exp_lines.append(f"{company}-{pos}{duration_str}{desc_str}")
            sections.append(f"工作经历：{'；'.join(exp_lines)}。")

        # 项目经历
        projects = cand.get("projects", [])
        if projects:
            proj_lines = []
            for proj in projects:
                proj_name = proj.get('project_name', '')
                role = proj.get('role', '')
                techs = proj.get("technologies", [])
                tech_str = ",".join(techs) if isinstance(techs, list) else str(techs)
                proj_lines.append(f"{proj_name}({tech_str})-{role}")
            sections.append(f"项目经历：{'；'.join(proj_lines)}。")

        return "\n".join(sections)
