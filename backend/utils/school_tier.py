"""权威院校分级工具

职责：
1. 以 data/knowledge/school_categories.json 为唯一权威数据源，对学校名称做层级分类。
2. 业务规则：中国科学院大学(国科大) 归入 985（已在 JSON 名单中维护）。
3. 提供分级权重 985 > 211 > 双一流 > 普通本科/专科，用于匹配排序加权。

设计原则：
- 入库阶段调用 normalize_school_tier() 强制覆盖 LLM 的主观判断（权威优先）。
- 不在任何权威名单中的学校：保留 LLM 兜底判断（fallback 参数）。
- 学校名做归一化匹配（去括号/空白，全称与简称兼容）。
- 海外院校使用具体排名数字（如 "QS 4"），而非区间标签（如 "QS Top 50"）。
"""
from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 权威知识库路径
_KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "school_categories.json"
# 海外院校排名知识库（QS 2025 + U.S. News 2024-2025 TOP150，映射到具体排名）
_OVERSEAS_KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "overseas_school_rankings.json"

# 标准层级（从高到低）
TIER_985 = "985"
TIER_211 = "211"
TIER_SHUANG = "双一流"
TIER_NORMAL = "普通本科"
TIER_ZHUANKE = "专科"
TIER_UNKNOWN = "unknown"

# 海外院校 QS 排名层级（保留旧常量用于兼容旧数据迁移/比较）
TIER_QS_TOP50 = "QS Top 50"
TIER_QS_TOP100 = "QS Top 100"
TIER_QS_TOP200 = "QS Top 200"
TIER_QS_TOP500 = "QS Top 500"

# 一流大学集合（985 + 211 + 双一流 + QS Top 200 以内），用于"一流大学"语义判断
TOP_TIER_SET = {TIER_985, TIER_211, TIER_SHUANG, TIER_QS_TOP50, TIER_QS_TOP100, TIER_QS_TOP200}

# 分级权重（985/QS Top 50 > 211/QS Top 100 > 双一流/QS Top 200 > 其它），数值越大越优。用于排序加权。
DEFAULT_TIER_RANK: Dict[str, int] = {
    TIER_985: 3,
    TIER_QS_TOP50: 3,
    TIER_211: 2,
    TIER_QS_TOP100: 2,
    TIER_SHUANG: 1,
    TIER_QS_TOP200: 1,
    TIER_QS_TOP500: 1,
    TIER_NORMAL: 0,
    TIER_ZHUANKE: 0,
    TIER_UNKNOWN: 0,
}

# 连续得分（用于和分数体系融合，0~1）
DEFAULT_TIER_SCORE: Dict[str, float] = {
    TIER_985: 1.0,
    TIER_QS_TOP50: 1.0,
    TIER_211: 0.85,
    TIER_QS_TOP100: 0.85,
    TIER_SHUANG: 0.80,
    TIER_QS_TOP200: 0.80,
    TIER_QS_TOP500: 0.70,
    TIER_NORMAL: 0.60,
    TIER_ZHUANKE: 0.40,
    TIER_UNKNOWN: 0.50,
}

# 正则：匹配 "QS N" 格式（如 "QS 4", "QS 127"）
_QS_RANK_RE = re.compile(r"^QS\s*(\d+)$")


def _parse_qs_rank(tier: Optional[str]) -> Optional[int]:
    """从 'QS N' 格式中提取排名数字，非此格式返回 None"""
    if not tier:
        return None
    m = _QS_RANK_RE.match(tier)
    return int(m.group(1)) if m else None


def _qs_rank_to_weight(rank: int) -> int:
    """将 QS 具体排名映射为权重（与国内层级对齐）"""
    if rank <= 50:
        return 3   # 等同 985
    elif rank <= 100:
        return 2   # 等同 211
    elif rank <= 200:
        return 1   # 等同 双一流
    else:
        return 1   # QS 201-500 也给 1


def _qs_rank_to_score(rank: int) -> float:
    """将 QS 具体排名映射为连续得分 0~1（排名越小分越高）"""
    if rank <= 10:
        return 1.0
    elif rank <= 50:
        return 0.95
    elif rank <= 100:
        return 0.85
    elif rank <= 200:
        return 0.80
    elif rank <= 500:
        return 0.70
    return 0.60


class SchoolTierClassifier:
    """院校层级权威分类器（单例使用）"""

    def __init__(self):
        self._tier_of: Dict[str, str] = {}          # 归一化校名 -> 层级
        self._tier_rank: Dict[str, int] = dict(DEFAULT_TIER_RANK)
        self._tier_score: Dict[str, float] = dict(DEFAULT_TIER_SCORE)
        self._loaded = False
        self._load()

    # ── 数据加载 ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not _KB_PATH.exists():
            logger.warning(f"[SchoolTier] 知识库不存在: {_KB_PATH}，分级将全部走 fallback")
            self._loaded = True
            return
        try:
            data = json.loads(_KB_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"[SchoolTier] 知识库解析失败: {e}")
            self._loaded = True
            return

        for tier in (TIER_985, TIER_211, TIER_SHUANG, TIER_NORMAL, TIER_ZHUANKE):
            for name in data.get(tier, []):
                self._tier_of[self._norm(name)] = tier

        meta = data.get("_metadata", {})
        if isinstance(meta.get("tier_rank"), dict):
            self._tier_rank.update(meta["tier_rank"])
        if isinstance(meta.get("tier_scoring"), dict):
            self._tier_score.update(meta["tier_scoring"])

        # 加载海外院校排名知识库（QS/U.S. News → 具体排名 "QS N"）
        self._load_overseas()

        self._loaded = True
        logger.info(f"[SchoolTier] 已加载 {len(self._tier_of)} 所院校层级映射（含海外名校）")

    @staticmethod
    def _rank_to_qs_tier(best_rank: int) -> Optional[str]:
        """根据 best_rank 生成具体排名标签，如 'QS 4'、'QS 127'"""
        if best_rank <= 0 or best_rank > 500:
            return None
        return f"QS {best_rank}"

    def _load_overseas(self) -> None:
        """加载海外院校排名知识库，将英文/中文校名映射到具体 QS 排名。

        简历中的海外校名通常为 'University of Oxford(牛津大学)' 形式，
        因此英文名、中文名分别归一化后都建立映射。
        使用 best_rank（QS 与 U.S. News 取更优排名）生成 "QS N" 格式标签。
        """
        if not _OVERSEAS_KB_PATH.exists():
            logger.info(f"[SchoolTier] 海外院校知识库不存在: {_OVERSEAS_KB_PATH}，跳过海外映射")
            return
        try:
            data = json.loads(_OVERSEAS_KB_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"[SchoolTier] 海外院校知识库解析失败: {e}")
            return

        count = 0
        for uni in data.get("universities", []):
            best_rank = uni.get("best_rank")
            if not isinstance(best_rank, int) or best_rank <= 0:
                continue
            tier = self._rank_to_qs_tier(best_rank)
            if tier is None:
                continue
            for key in ("name_en", "name_cn"):
                name = uni.get(key)
                norm = self._norm(name)
                # 不覆盖国内权威名单中已存在的同名（如"东北大学"中外重名以国内为准）
                if norm and norm not in self._tier_of:
                    self._tier_of[norm] = tier
                    count += 1
        logger.info(f"[SchoolTier] 已加载 {count} 条海外院校 QS 排名映射")

    @staticmethod
    def _norm(name: Optional[str]) -> str:
        """归一化校名：去空白、去括号内容、去常见后缀噪声"""
        if not name:
            return ""
        s = str(name).strip()
        # 去掉括号及其中内容（如"四川大学(双一流)"）
        s = re.sub(r"[（(\[【].*?[）)\]】]", "", s)
        s = re.sub(r"\s+", "", s)
        return s

    # ── 核心：分类 ────────────────────────────────────────────────────────
    def classify(self, school_name: Optional[str], fallback: Optional[str] = None) -> str:
        """返回学校的权威层级。

        优先级：
        1. 精确匹配权威名单 -> 返回权威层级
        2. 包含匹配（处理别名/全简称差异）-> 返回权威层级
        3. 不在名单 -> 返回 fallback（若为合法层级），否则 unknown
        """
        norm = self._norm(school_name)
        if not norm:
            return self._valid_fallback(fallback)

        # 1. 精确匹配
        if norm in self._tier_of:
            return self._tier_of[norm]

        # 2. 包含匹配（候选校名包含权威校名，或反之）
        for auth_name, tier in self._tier_of.items():
            if auth_name and (auth_name in norm or norm in auth_name) and len(auth_name) >= 3:
                return tier

        # 3. 兜底
        return self._valid_fallback(fallback)

    def _valid_fallback(self, fallback: Optional[str]) -> str:
        """校验 fallback 是否为合法层级，非法则归 unknown。
        支持旧格式 'QS Top 50' 和新格式 'QS 4'。
        """
        if not fallback:
            return TIER_UNKNOWN
        # 国内层级
        domestic_tiers = (TIER_985, TIER_211, TIER_SHUANG, TIER_NORMAL, TIER_ZHUANKE)
        if fallback in domestic_tiers:
            return fallback
        # 旧格式 QS Top N（兼容）
        old_qs_tiers = (TIER_QS_TOP50, TIER_QS_TOP100, TIER_QS_TOP200, TIER_QS_TOP500)
        if fallback in old_qs_tiers:
            return fallback
        # 新格式 QS N
        if _parse_qs_rank(fallback) is not None:
            return fallback
        return TIER_UNKNOWN

    def normalize_tier(self, school_name: Optional[str], llm_tier: Optional[str] = None) -> str:
        """入库归一化入口：用权威名单覆盖 LLM 判断，名单外保留 LLM 兜底。"""
        return self.classify(school_name, fallback=llm_tier)

    # ── 权重/得分 ─────────────────────────────────────────────────────────
    def tier_rank(self, tier: Optional[str]) -> int:
        """层级排序权重：985=3 > 211=2 > 双一流=1 > 其它=0
        支持 'QS N' 格式动态计算权重。
        """
        if not tier:
            return 0
        # 先查静态表
        if tier in self._tier_rank:
            return self._tier_rank[tier]
        # 动态处理 "QS N" 格式
        rank = _parse_qs_rank(tier)
        if rank is not None:
            return _qs_rank_to_weight(rank)
        return 0

    def tier_score(self, tier: Optional[str]) -> float:
        """层级连续得分 0~1
        支持 'QS N' 格式动态计算得分。
        """
        if not tier:
            return self._tier_score.get(TIER_UNKNOWN, 0.5)
        # 先查静态表
        if tier in self._tier_score:
            return self._tier_score[tier]
        # 动态处理 "QS N" 格式
        rank = _parse_qs_rank(tier)
        if rank is not None:
            return _qs_rank_to_score(rank)
        return self._tier_score.get(TIER_UNKNOWN, 0.5)

    def is_top_tier(self, tier: Optional[str]) -> bool:
        """是否一流大学（985/211/双一流/QS Top 200 以内）"""
        if tier in TOP_TIER_SET:
            return True
        # 动态判断 "QS N" 格式
        rank = _parse_qs_rank(tier)
        if rank is not None:
            return rank <= 200
        return False

    def best_tier_of_candidate(self, education_history: List[Dict]) -> Optional[str]:
        """给定候选人教育经历，返回其学历中最高的院校层级（按权威重判每段）。"""
        best_tier = None
        best_rank = -1
        for edu in education_history or []:
            if not isinstance(edu, dict):
                continue
            tier = self.classify(edu.get("school"), fallback=edu.get("school_tier"))
            r = self.tier_rank(tier)
            if r > best_rank:
                best_rank = r
                best_tier = tier
        return best_tier


# 全局单例
school_tier_classifier = SchoolTierClassifier()


# 便捷函数
def normalize_school_tier(school_name: Optional[str], llm_tier: Optional[str] = None) -> str:
    """入库时归一化院校层级（权威优先，名单外保留 LLM 兜底）"""
    return school_tier_classifier.normalize_tier(school_name, llm_tier)


def get_tier_rank(tier: Optional[str]) -> int:
    return school_tier_classifier.tier_rank(tier)


def get_tier_score(tier: Optional[str]) -> float:
    return school_tier_classifier.tier_score(tier)


def is_top_tier(tier: Optional[str]) -> bool:
    return school_tier_classifier.is_top_tier(tier)
