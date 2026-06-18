"""候选人类别定义模块 - 应届生/实习生/社招的动态判定

核心定义（基于当前北京时间动态计算）：

1. 应届生（Fresh Graduate）:
   - 今年或去年毕业的候选人
   - 例：当前2026年6月 → 2025届和2026届均为应届生
   - 时间窗口：毕业年份 ∈ [current_year - 1, current_year]
   - 工作年限：通常 0-2 年

2. 实习生（Intern）:
   - 尚未毕业的在校学生
   - 毕业年份 > 当前年份（或当前年份但还未到毕业月份6月）
   - 例：当前2026年6月 → 2027届及之后 = 实习生
   - 工作年限：通常 0-1 年

3. 社招（Experienced/Social Recruitment）:
   - 毕业时间 < current_year - 1，已有正式工作经验
   - 不属于以上两类的候选人

毕业月份假设：
- 中国高校毕业时间通常为6月（7月1日前离校）
- 因此"2026届"在2026年7月之前仍为在校生
"""

import logging
from typing import Dict, Any, Optional, Tuple
from backend.utils.time_utils import get_current_year, get_current_month

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 核心定义常量
# ═══════════════════════════════════════════════════════════════════════════════

# 中国高校典型毕业月份
GRADUATION_MONTH = 6  # 6月毕业（7月前离校）


def get_fresh_grad_year_range() -> Tuple[int, int]:
    """获取应届生毕业年份范围

    规则：当前年份和去年都算应届
    例：2026年 → (2025, 2026) 都是应届生

    Returns:
        (min_year, max_year) 应届生毕业年份闭区间
    """
    current_year = get_current_year()
    return (current_year - 1, current_year)


def get_intern_grad_year_min() -> int:
    """获取实习生（在校生）的最小毕业年份

    实习生 = 距离毕业还有至少一年的在校生。
    
    规则：
    - 实习生应该是明年及之后毕业的学生（current_year + 1 及以后）
    - 今年毕业的算"应届生"不算"实习生"
    
    例：2026年6月 → 实习生 = 2027届及之后（今年的是应届生）
        2026年9月 → 实习生 = 2027届及之后

    Returns:
        实习生的最小毕业年份
    """
    current_year = get_current_year()
    return current_year + 1


def classify_candidate(grad_year: Optional[int], work_years: int = 0) -> str:
    """根据毕业年份和工作年限判断候选人类别

    Args:
        grad_year: 最高学历毕业年份（None表示未知）
        work_years: 工作年限

    Returns:
        "intern" | "fresh_grad" | "experienced" | "unknown"
    """
    if grad_year is None:
        # 没有毕业年份信息，靠工作年限粗判
        if work_years <= 1:
            return "fresh_grad"  # 保守归为应届
        return "experienced"

    intern_min = get_intern_grad_year_min()
    fresh_min, fresh_max = get_fresh_grad_year_range()

    if grad_year >= intern_min:
        # 还没毕业
        return "intern"
    elif fresh_min <= grad_year <= fresh_max:
        return "fresh_grad"
    else:
        return "experienced"


def get_max_work_years_for_category(category: str) -> int:
    """获取某类候选人的最大合理工作年限

    Args:
        category: "intern" | "fresh_grad" | "experienced"

    Returns:
        最大工作年限
    """
    if category == "intern":
        return 1
    elif category == "fresh_grad":
        return 2
    else:
        return 9999


def get_grad_year_filter_for_query(query: str) -> Dict[str, Any]:
    """根据用户查询解析应届/实习/毕业年份过滤条件

    动态计算，不硬编码年份。

    Args:
        query: 用户自然语言查询

    Returns:
        过滤条件字典，可能包含:
        - is_intern: bool
        - is_fresh_grad: bool
        - max_work_years: int
        - grad_year_min: int
        - grad_year_max: int
    """
    import re

    filters: Dict[str, Any] = {}
    current_year = get_current_year()

    # 检测实习生关键词
    if re.search(r'实习[生]?|在校生?|日常实习', query):
        filters["is_intern"] = True
        filters["max_work_years"] = 1
        # 实习生：毕业年份 >= intern_min
        filters["grad_year_min"] = get_intern_grad_year_min()
        filters["grad_year_max"] = 9999
        logger.info(f"[Category] Detected intern query, grad_year >= {filters['grad_year_min']}")
        return filters

    # 检测应届生关键词
    if re.search(r'应届[生毕业]*|今年毕业|刚毕业', query):
        fresh_min, fresh_max = get_fresh_grad_year_range()
        filters["is_fresh_grad"] = True
        filters["max_work_years"] = 2
        filters["grad_year_min"] = fresh_min
        filters["grad_year_max"] = fresh_max
        logger.info(f"[Category] Detected fresh grad query, grad_year in [{fresh_min}, {fresh_max}]")
        return filters

    # 检测"X届"表达
    grad_match = re.search(r'(\d{4})\s*届', query)
    if grad_match:
        target_year = int(grad_match.group(1))
        intern_min = get_intern_grad_year_min()
        fresh_min, fresh_max = get_fresh_grad_year_range()

        if target_year >= intern_min:
            # 目标届还没毕业 → 实习生
            filters["is_intern"] = True
            filters["max_work_years"] = 1
            filters["grad_year_min"] = target_year
            filters["grad_year_max"] = target_year
        elif fresh_min <= target_year <= fresh_max:
            # 目标届刚毕业 → 应届生
            filters["is_fresh_grad"] = True
            filters["max_work_years"] = 2
            filters["grad_year_min"] = target_year
            filters["grad_year_max"] = target_year
        else:
            # 往届 → 社招
            filters["grad_year_min"] = target_year
            filters["grad_year_max"] = target_year
        logger.info(f"[Category] Detected '{target_year}届', filters={filters}")
        return filters

    # 检测"今年毕业"
    if re.search(r'今年毕业|今年[的]?毕业生', query):
        filters["is_fresh_grad"] = True
        filters["max_work_years"] = 2
        filters["grad_year_min"] = current_year
        filters["grad_year_max"] = current_year
        return filters

    return filters


def build_category_knowledge() -> str:
    """生成当前时间下的应届生/实习生定义知识文本（可注入RAG/Prompt）

    Returns:
        描述当前定义的自然语言文本
    """
    current_year = get_current_year()
    current_month = get_current_month()
    fresh_min, fresh_max = get_fresh_grad_year_range()
    intern_min = get_intern_grad_year_min()

    knowledge = f"""【候选人类别定义（动态，基于当前时间 {current_year}年{current_month}月）】

1. 应届毕业生：毕业年份为 {fresh_min} 年或 {fresh_max} 年的候选人。即去年和今年毕业的学生。
   - 工作年限通常 0-2 年
   - 搜索"应届生"时应筛选这两届

2. 实习生/在校生：毕业年份 >= {intern_min} 年的候选人（尚未正式毕业）。
   - 工作年限通常 0-1 年（仅有实习经历）
   - {intern_min}届及之后的学生当前仍在校

3. 社招/有经验候选人：毕业年份 < {fresh_min} 年的候选人。
   - 已有正式工作经验

注：中国高校毕业季为每年6月。{current_year}届学生在{current_year}年7月前仍为在校生。"""
    return knowledge
