"""候选人字段归一化工具。

LLM 抽取出的 job_status / education_level 往往是高度自由的自然语言
（如"离职，可立即到岗"、"硕士（在读）"、"应届硕士在读，寻求实习岗位"），
不利于后续按枚举精确过滤。本模块把这些自由文本映射到稳定枚举值。

设计原则：
- 归一化只改写 candidates 表用于过滤的主字段；不丢失语义（在读/非全日制等
  细节仍由 education_history.is_fulltime、end_date 等承载）。
- 映射不到时，返回 UNKNOWN 占位（job_status）或保守保留（education_level），
  绝不抛异常，保证入库链路稳定。
"""

import re
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# 求职状态枚举
# ─────────────────────────────────────────────────────────────────────────────
JS_ON_LOOKING = "在职看机会"
JS_ON_NOT_LOOKING = "在职不看"
JS_LEFT = "离职"
JS_FRESH = "应届毕业生"
JS_INTERN = "实习"
JS_STUDYING = "在读"
JS_SEEKING = "求职中"
JS_UNKNOWN = "未知"

JOB_STATUS_ENUM = [
    JS_ON_LOOKING, JS_ON_NOT_LOOKING, JS_LEFT, JS_FRESH,
    JS_INTERN, JS_STUDYING, JS_SEEKING, JS_UNKNOWN,
]


def normalize_job_status(raw: Optional[str]) -> str:
    """把自由文本求职状态映射为枚举值。

    判定按优先级匹配关键词；同一文本可能含多义（如"应届硕士在读，寻求实习"），
    按"实习 > 应届 > 在读 > 离职 > 在职(看/不看) > 求职中"的业务优先级裁决。
    """
    if not raw or not str(raw).strip():
        return JS_UNKNOWN
    s = str(raw).strip()

    # 实习（含"寻求实习/实习生/可全职实习"等）
    if re.search(r"实习", s):
        return JS_INTERN
    # 应届
    if re.search(r"应届|\d{4}\s*届|刚毕业|今年毕业", s):
        return JS_FRESH
    # 在读 / 在校（无实习/应届修饰）
    if re.search(r"在读|在校|备战秋招|学习阶段", s):
        return JS_STUDYING
    # 离职（含"已离职/离职状态/可立即到岗"等）
    if re.search(r"离职|已离职|待业", s):
        return JS_LEFT
    # 在职：区分看/不看
    if re.search(r"在职", s):
        if re.search(r"不看|暂不考虑|不考虑|暂不", s):
            return JS_ON_NOT_LOOKING
        if re.search(r"看机会|看看|考虑|机会|交流", s):
            return JS_ON_LOOKING
        # 仅"在职"无倾向 → 默认看机会（最常见招聘可触达状态）
        return JS_ON_LOOKING
    # 单纯求职/可到岗
    if re.search(r"求职|可立即|可随时|可快速|可尽快|入职|到岗|找工作", s):
        return JS_SEEKING

    return JS_UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# 学历层级枚举
# ─────────────────────────────────────────────────────────────────────────────
EDU_DOCTOR = "博士"
EDU_MASTER = "硕士"
EDU_BACHELOR = "本科"
EDU_COLLEGE = "专科"
EDU_OTHER = "其他"

EDUCATION_ENUM = [EDU_DOCTOR, EDU_MASTER, EDU_BACHELOR, EDU_COLLEGE, EDU_OTHER]


def normalize_education_level(raw: Optional[str], fallback: Optional[str] = None) -> str:
    """把自由文本学历映射为标准学历层级。

    去掉"在读/非全日制/研究生/全日制"等修饰，归一到最高学历层级本身。
    在读/非全日制等细节不在此字段承载（由 education_history 体现）。
    映射不到时：若给了 fallback 则返回 fallback，否则 EDU_OTHER。
    """
    if not raw or not str(raw).strip():
        return fallback if fallback else EDU_OTHER
    s = str(raw).strip()

    # 博士（含博士后归博士）
    if re.search(r"博士|博士后|phd|doctor", s, re.I):
        return EDU_DOCTOR
    # 硕士 / 研究生
    if re.search(r"硕士|研究生|master|mba|工程硕士", s, re.I):
        return EDU_MASTER
    # 本科（含学士、专升本的最终学历为本科）
    if re.search(r"本科|学士|专升本|bachelor", s, re.I):
        return EDU_BACHELOR
    # 专科 / 大专 / 高职
    if re.search(r"专科|大专|高职|高专", s, re.I):
        return EDU_COLLEGE

    return fallback if fallback else EDU_OTHER
