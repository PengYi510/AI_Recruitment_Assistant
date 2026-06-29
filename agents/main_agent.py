"""主Agent - 多轮对话管理与记忆增强查询处理

核心职责:
1. 短期记忆管理: 维护对话历史中的关键上下文（实体、意图、结果摘要）
2. 查询改写: 对多轮指代/省略表达进行上下文增强，还原完整语义
3. 交互续接: 处理挂起的用户交互（确认/选择/输入）
4. 调度执行: 将增强后的查询交给 Harness 流程处理

多轮对话示例:
  用户: "帮我找Java高级工程师"
  系统: [返回候选人列表]
  用户: "看看第一个候选人的详细信息"
  → 查询改写: "查看候选人001的详细信息" (通过短期记忆解析"第一个"的指代)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── 短期记忆数据结构 ────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """单轮对话的记忆条目

    存储每轮对话中需要被后续轮次引用的关键信息:
    - 用户原始查询与改写后查询
    - 识别出的实体（人名、部门、职级等）
    - 查询意图
    - 结果摘要（如返回的候选人列表，供后续指代消解）
    """
    turn_id: int
    user_query: str
    rewritten_query: str
    intent: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    result_items: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ShortTermMemory:
    """短期记忆模块

    功能:
    - 维护最近 N 轮对话的结构化记忆（默认保留最近10轮）
    - 支持实体追踪（跨轮次维护当前讨论的实体）
    - 支持结果引用（记录上一轮返回的列表型结果，用于序号指代消解）

    设计原则:
    - 仅在内存中维护（由 SessionStore 负责持久化到 Redis/SQLite）
    - 记忆条目的序列化/反序列化通过 to_dict/from_dict 支持
    - 与 http_server.py 的 history 字段配合使用
    """

    MAX_TURNS = 10  # 最大保留轮次

    def __init__(self):
        self.entries: List[MemoryEntry] = []
        self.active_entities: Dict[str, Any] = {}  # 当前活跃实体（跨轮次累积）
        self.last_result_items: List[Dict[str, Any]] = []  # 上一轮返回的列表型结果

    def add_entry(self, entry: MemoryEntry) -> None:
        """添加一轮记忆"""
        self.entries.append(entry)
        # 滑动窗口：超出最大轮次时移除最早的记忆
        if len(self.entries) > self.MAX_TURNS:
            self.entries = self.entries[-self.MAX_TURNS:]
        # 更新活跃实体
        if entry.entities:
            self.active_entities.update(entry.entities)
        # 更新最新的列表型结果
        if entry.result_items:
            self.last_result_items = entry.result_items

    def get_recent_context(self, n: int = 3) -> str:
        """获取最近 n 轮对话的摘要上下文，用于查询改写的 prompt"""
        if not self.entries:
            return ""
        recent = self.entries[-n:]
        context_parts = []
        for entry in recent:
            part = f"[第{entry.turn_id}轮] 用户: {entry.user_query}"
            if entry.intent:
                part += f" | 意图: {entry.intent}"
            if entry.result_summary:
                part += f" | 结果: {entry.result_summary}"
            context_parts.append(part)
        return "\n".join(context_parts)

    def get_active_entities_str(self) -> str:
        """获取当前活跃实体的描述字符串"""
        if not self.active_entities:
            return ""
        parts = [f"{k}: {v}" for k, v in self.active_entities.items()]
        return "; ".join(parts)

    def resolve_reference(self, query: str) -> Optional[str]:
        """尝试解析序号指代

        当用户说"第一个"、"第二个"等序号指代时，从 last_result_items 中解析真实实体。
        返回解析后的候选人名称或 None（无法解析时）。

        示例:
            last_result_items = [{"name": "候选人001"}, {"name": "候选人002"}]
            query = "看看第一个候选人的详细信息"
            → 返回 "候选人001"
        """
        if not self.last_result_items:
            return None

        # 中文序号映射
        ordinal_map = {
            "第一": 0, "第二": 1, "第三": 2, "第四": 3, "第五": 4,
            "第六": 5, "第七": 6, "第八": 7, "第九": 8, "第十": 9,
            "第1": 0, "第2": 1, "第3": 2, "第4": 3, "第5": 4,
            "第6": 5, "第7": 6, "第8": 7, "第9": 8, "第10": 9,
        }
        for ordinal, idx in ordinal_map.items():
            if ordinal in query and idx < len(self.last_result_items):
                item = self.last_result_items[idx]
                return item.get("name") or item.get("id") or str(item)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于存入 SessionStore）"""
        return {
            "entries": [
                {
                    "turn_id": e.turn_id,
                    "user_query": e.user_query,
                    "rewritten_query": e.rewritten_query,
                    "intent": e.intent,
                    "entities": e.entities,
                    "result_summary": e.result_summary,
                    "result_items": e.result_items,
                    "timestamp": e.timestamp,
                }
                for e in self.entries
            ],
            "active_entities": self.active_entities,
            "last_result_items": self.last_result_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShortTermMemory":
        """从字典反序列化（从 SessionStore 恢复）"""
        memory = cls()
        if not data:
            return memory
        for entry_dict in data.get("entries", []):
            entry = MemoryEntry(
                turn_id=entry_dict["turn_id"],
                user_query=entry_dict["user_query"],
                rewritten_query=entry_dict["rewritten_query"],
                intent=entry_dict.get("intent", ""),
                entities=entry_dict.get("entities", {}),
                result_summary=entry_dict.get("result_summary", ""),
                result_items=entry_dict.get("result_items", []),
                timestamp=entry_dict.get("timestamp", 0),
            )
            memory.entries.append(entry)
        memory.active_entities = data.get("active_entities", {})
        memory.last_result_items = data.get("last_result_items", [])
        return memory


# ── 查询改写模块 ────────────────────────────────────────────────────────────────

class QueryRewriter:
    """查询改写模块

    对多轮对话中的省略、指代、上下文依赖表达进行改写，还原完整查询语义。

    改写策略:
    1. 指代消解: "第一个"、"那个人"等 → 解析为具体实体名称
    2. 省略补全: "换一个部门呢？" → 补全前文提到的筛选条件
    3. 上下文融合: 结合历史对话意图，丰富当前查询的语义

    当前实现: 基于规则的改写（适用于论文架构验证）
    生产版本: 可替换为 LLM-based 查询改写（输入 context + query → 输出 rewritten_query）
    """

    # 需要上下文才能理解的指代词/省略模式
    REFERENCE_PATTERNS = [
        "第一", "第二", "第三", "第四", "第五",
        "第1", "第2", "第3", "第4", "第5",
        "上一个", "下一个", "这个", "那个",
        "他", "她", "这个人", "那个人",
    ]

    CONTINUATION_PATTERNS = [
        "还有呢", "继续", "更多", "其他的",
        "换一个", "再来一个", "下一页",
    ]

    # 表示省略/承接前文的语言模式（"那就"、"改成"等）
    ELLIPSIS_PATTERNS = [
        "那就", "那找", "那搜", "改成", "换成",
        "调整为", "改为", "放宽到", "放宽为",
        "以上", "以下", "年以上", "年以下",
    ]

    def needs_rewrite(self, query: str, memory: ShortTermMemory) -> bool:
        """判断当前查询是否需要改写

        条件:
        - 存在历史记忆（非首轮对话）
        - 查询中包含指代词/省略表达
        - 或查询过短（可能是省略表达）
        - 或查询包含承接前文的语言模式（"那就"、"年以上"等）
        """
        if not memory.entries:
            return False
        # 包含指代模式
        for pattern in self.REFERENCE_PATTERNS + self.CONTINUATION_PATTERNS:
            if pattern in query:
                return True
        # 包含省略/承接模式（暗示需要从上下文补全信息）
        for pattern in self.ELLIPSIS_PATTERNS:
            if pattern in query:
                return True
        # 查询过短且有历史上下文（可能是省略表达）
        if len(query) <= 15 and memory.entries:
            return True
        return False

    def rewrite(self, query: str, memory: ShortTermMemory) -> str:
        """执行查询改写（LLM-based + 规则回退）

        改写流程:
        1. 尝试序号指代消解（"第一个" → 具体候选人名称）
        2. 尝试续接模式补全（"还有呢" → 重复上一轮的查询条件+分页）
        3. 使用 LLM 进行上下文感知的查询改写（核心策略）
        4. 规则式回退：实体省略补全

        Args:
            query: 用户原始查询
            memory: 当前短期记忆

        Returns:
            改写后的完整查询（如无需改写则返回原始查询）
        """
        rewritten = query

        # 策略1: 序号指代消解
        resolved_name = memory.resolve_reference(query)
        if resolved_name:
            # 将序号表达替换为具体名称
            for pattern in self.REFERENCE_PATTERNS:
                if pattern in rewritten:
                    rewritten = rewritten.replace(
                        f"{pattern}个候选人", resolved_name
                    ).replace(
                        f"{pattern}个", resolved_name
                    ).replace(
                        f"{pattern}位", resolved_name
                    )
                    break
            logger.info(f"[QueryRewrite] 指代消解: '{query}' → '{rewritten}'")
            return rewritten

        # 策略2: 续接模式（"还有呢"、"继续"等 → 翻页或继续上一轮查询）
        for pattern in self.CONTINUATION_PATTERNS:
            if pattern in query and memory.entries:
                last_entry = memory.entries[-1]
                rewritten = f"{last_entry.rewritten_query}（继续/下一页）"
                logger.info(f"[QueryRewrite] 续接改写: '{query}' → '{rewritten}'")
                return rewritten

        # 策略3: LLM-based 上下文感知查询改写
        llm_rewritten = self._llm_rewrite(query, memory)
        if llm_rewritten and llm_rewritten != query:
            logger.info(f"[QueryRewrite] LLM改写: '{query}' → '{llm_rewritten}'")
            return llm_rewritten

        # 策略4: 规则式回退 - 实体省略补全
        if memory.active_entities:
            entities_str = memory.get_active_entities_str()
            if entities_str:
                rewritten = f"{query}（上下文: {entities_str}）"
                logger.info(f"[QueryRewrite] 实体补全: '{query}' → '{rewritten}'")
                return rewritten

        return rewritten

    def _llm_rewrite(self, query: str, memory: ShortTermMemory) -> Optional[str]:
        """使用 LLM 进行上下文感知的查询改写

        将对话历史和当前查询送入 LLM，让其生成完整的、不依赖上下文的独立查询。
        """
        from backend.models.longcat_client import chat_completion

        context = memory.get_recent_context()
        entities_str = memory.get_active_entities_str()

        system_prompt = """你是一个查询改写专家。你的任务是将用户的多轮对话中的省略、指代表达改写为一个完整的、独立的查询语句。

规则：
1. 结合对话历史，补全当前查询中省略的信息（如职位类型、技能要求等）
2. 改写后的查询必须是独立完整的，不依赖任何上下文就能理解
3. 保留用户最新一轮的核心意图（如调整条件、放宽要求等）
4. 只输出改写后的查询文本，不要解释
5. 如果当前查询已经是完整的，则原样返回"""

        user_prompt = f"""对话历史：
{context}

当前活跃实体：{entities_str}

用户当前输入：{query}

请将用户当前输入改写为完整的独立查询："""

        try:
            response = chat_completion(system=system_prompt, user=user_prompt, temperature=0.1)
            result = (response.content or "").strip()
            # 去除可能的引号包裹
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            if result.startswith("'") and result.endswith("'"):
                result = result[1:-1]
            return result if result else None
        except Exception as e:
            logger.warning(f"[QueryRewrite] LLM改写失败: {e}")
            return None


# ── 意图预分类模块 ──────────────────────────────────────────────────────────────

class IntentClassifier:
    """意图预分类器

    在进入 Harness 流程之前，快速判断用户输入属于哪种类型：
    1. chitchat  - 闲聊/打招呼/无关话题（如"你好"、"1"、"天气"）
    2. system_query - 系统信息查询（如"简历库有多少人"、"学历分布"、"介绍一下你自己"）
    3. recruitment - 招聘相关查询（如"帮我找Java高级工程师"、完整JD等）

    分类策略：先用规则快速判断，无法确定时再调 LLM。
    """

    # 明确的招聘意图关键词
    RECRUITMENT_KEYWORDS = [
        "找人", "帮我找", "推荐", "搜索候选人", "匹配", "筛选",
        "招聘", "招人", "候选人", "简历",
        "工程师", "产品经理", "设计师", "分析师", "架构师", "运营",
        "开发", "算法", "后端", "前端", "全栈", "测试",
        "岗位职责", "任职要求", "岗位要求", "职位要求",
        "工作经验", "学历要求", "薪资", "年薪",
        "JD", "岗位描述",
    ]

    # 系统查询关键词/模式
    SYSTEM_QUERY_PATTERNS = [
        r"简历库.*(多少|数量|几个|统计|分布|占比|比例|概况|总览)",
        r"(候选人|人员).*(多少|数量|几个|统计|分布|占比|比例|概况|总览)",
        r"(学历|年龄|性别|城市|技能|院校|薪资|工作年限).*(分布|统计|占比|概况|情况)",
        r"(有多少|共有|总共|一共).*(候选人|简历|人)",
        r"(数据库|数据|库里).*(多少|什么|哪些|统计|概况)",
        r"介绍一下(你自己|你|系统|这个系统)",
        r"你是(谁|什么|哪个|做什么的)",
        r"(系统|你)(能做什么|有什么功能|怎么用|怎么使用|有什么能力)",
        r"帮助|help|使用说明|使用方法",
    ]

    # 明确的闲聊模式
    CHITCHAT_PATTERNS = [
        r"^(你好|hello|hi|hey|嗨|哈喽|嘿)\s*[!！。.]*$",
        r"^(谢谢|thanks|thank you|感谢|多谢)\s*[!！。.]*$",
        r"^(好的|好|ok|OK|行|嗯|明白了|了解)\s*[!！。.]*$",
        r"^(再见|bye|拜拜|下次见)\s*[!！。.]*$",
        r"^[0-9]+$",
        r"^.{0,2}$",
        r"天气|时间|几点|日期",
        r"讲个笑话|开心|无聊",
        r"^(测试|test)\s*[!！。.]*$",
    ]

    def classify(self, query: str, memory: "ShortTermMemory") -> str:
        """对用户输入进行意图分类

        Returns:
            "chitchat" | "system_query" | "recruitment"
        """
        import re

        query_stripped = query.strip()

        # ── 规则1: 如果有短期记忆中的指代/续接，仍走招聘流程 ──
        if memory.entries:
            for pattern in QueryRewriter.REFERENCE_PATTERNS + QueryRewriter.CONTINUATION_PATTERNS:
                if pattern in query_stripped:
                    return "recruitment"

        # ── 规则2: 明确的闲聊模式 ──
        for pattern in self.CHITCHAT_PATTERNS:
            if re.search(pattern, query_stripped, re.IGNORECASE):
                return "chitchat"

        # ── 规则3: 系统信息查询模式 ──
        for pattern in self.SYSTEM_QUERY_PATTERNS:
            if re.search(pattern, query_stripped, re.IGNORECASE):
                return "system_query"

        # ── 规则4: 包含明确招聘关键词 ──
        for keyword in self.RECRUITMENT_KEYWORDS:
            if keyword in query_stripped:
                return "recruitment"

        # ── 规则5: 长文本（可能是JD）→ 招聘 ──
        if len(query_stripped) > 80:
            return "recruitment"

        # ── 规则6: 中等长度但无明确意图 → 调 LLM 判断 ──
        return self._llm_classify(query_stripped)

    def _llm_classify(self, query: str) -> str:
        """使用 LLM 进行意图分类（兜底方案）"""
        from backend.models.longcat_client import chat_completion

        system_prompt = """你是一个意图分类器。判断用户输入属于以下哪种类型，只返回类别名称：

1. chitchat - 闲聊、打招呼、无关话题、测试性输入
2. system_query - 询问系统本身的信息（如系统功能介绍、数据库统计信息、候选人总量/分布等）
3. recruitment - 与招聘相关的查询（搜索候选人、职位匹配、简历筛选等）

只返回一个词：chitchat 或 system_query 或 recruitment"""

        try:
            response = chat_completion(system=system_prompt, user=query, temperature=0.0, max_tokens=20)
            result = (response.content or "").strip().lower()
            if result in ("chitchat", "system_query", "recruitment"):
                return result
            return "recruitment"
        except Exception as e:
            logger.warning(f"[IntentClassifier] LLM 分类失败: {e}, 默认走 recruitment")
            return "recruitment"


# ── 系统查询处理模块 ──────────────────────────────────────────────────────────────

def _handle_system_query(query: str, emp_id: str) -> Dict[str, Any]:
    """处理系统信息类查询（如"简历库有多少人"、"学历分布"、"介绍一下你自己"等）"""
    from backend.models.longcat_client import chat_completion
    from backend.database.models import hr_db

    stats = _get_system_stats()

    system_prompt = """你是「AIBP 智能人力助手」，一个专业的AI招聘匹配系统。

你的核心能力：
- 智能简历匹配：根据JD或自然语言描述，从简历库中检索和匹配最合适的候选人
- SHAP可解释性：为每次匹配提供透明的评分解释，说明为什么推荐某位候选人
- 多轮对话：支持上下文理解、指代消解和条件追加
- 个性化记忆：记住用户的偏好和招聘习惯

当前系统数据概况：
""" + stats + """

回答规则：
1. 友好、专业、简洁
2. 如果用户问数据统计相关问题，基于上面的系统数据进行回答
3. 如果用户问"你是谁/介绍自己"等，介绍系统能力
4. 使用 Markdown 格式，但不要过度格式化
5. 在回复末尾可以给出一些引导性建议，帮助用户更好地使用系统"""

    try:
        response = chat_completion(system=system_prompt, user=query, temperature=0.3)
        answer = (response.content or "").strip()
        if answer:
            return {
                "answer": answer,
                "suggestions": ["帮我找Java高级工程师", "查看学历分布", "推荐有5年经验的算法工程师"],
            }
    except Exception as e:
        logger.warning(f"[SystemQuery] LLM 回答失败: {e}")

    return {
        "answer": "我是 AIBP 智能人力助手，可以帮您智能匹配候选人。试试输入岗位要求或描述您想找的人才吧！",
        "suggestions": ["帮我找Java高级工程师", "推荐有5年经验的算法工程师"],
    }


def _handle_chitchat(query: str, emp_id: str) -> Dict[str, Any]:
    """处理闲聊/无关对话，直接用 LLM 生成自然语言回复"""
    from backend.models.longcat_client import chat_completion

    system_prompt = """你是「AIBP 智能人力助手」，一个专业友好的AI招聘匹配系统。

当用户进行闲聊或非招聘相关对话时，你应该：
1. 友好地回应用户
2. 简短自然，不要冗长
3. 适当引导用户使用系统的核心功能（候选人搜索和匹配）
4. 不要生硬地拒绝对话，保持自然的交流感

你的核心功能是帮助HR智能匹配和搜索候选人，如果用户没有明确的招聘需求，可以友好回应后简单提示一下你能帮忙做什么。"""

    try:
        response = chat_completion(system=system_prompt, user=query, temperature=0.5, max_tokens=300)
        answer = (response.content or "").strip()
        if answer:
            return {
                "answer": answer,
                "suggestions": ["帮我找Java高级工程师", "简历库有多少候选人", "推荐算法工程师"],
            }
    except Exception as e:
        logger.warning(f"[Chitchat] LLM 回答失败: {e}")

    return {
        "answer": "你好！我是 AIBP 智能人力助手，可以帮您搜索和匹配候选人。有什么招聘需求可以告诉我～",
        "suggestions": ["帮我找Java高级工程师", "简历库有多少候选人", "推荐算法工程师"],
    }


def _get_system_stats() -> str:
    """从数据库获取系统统计概况"""
    from backend.database.models import hr_db

    try:
        with hr_db._get_conn() as conn:
            cursor = conn.cursor()

            # 候选人总数
            cursor.execute("SELECT COUNT(*) FROM candidates")
            total = cursor.fetchone()[0]

            # 检测学历字段名（兼容 education_level / highest_education）
            cursor.execute("PRAGMA table_info(candidates)")
            col_names = [row[1] for row in cursor.fetchall()]
            edu_col = "highest_education" if "highest_education" in col_names else "education_level"

            # 学历分布
            cursor.execute(f"""
                SELECT {edu_col}, COUNT(*) as cnt 
                FROM candidates 
                WHERE {edu_col} IS NOT NULL AND {edu_col} != ''
                GROUP BY {edu_col} 
                ORDER BY cnt DESC
            """)
            edu_dist = [(row[0], row[1]) for row in cursor.fetchall()]

            # 性别分布
            cursor.execute("""
                SELECT gender, COUNT(*) as cnt 
                FROM candidates 
                WHERE gender IS NOT NULL AND gender != ''
                GROUP BY gender
            """)
            gender_dist = [(row[0], row[1]) for row in cursor.fetchall()]

            # 年龄分布（分段）
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN age < 25 THEN '25岁以下'
                        WHEN age BETWEEN 25 AND 30 THEN '25-30岁'
                        WHEN age BETWEEN 31 AND 35 THEN '31-35岁'
                        WHEN age BETWEEN 36 AND 40 THEN '36-40岁'
                        ELSE '40岁以上'
                    END as age_range,
                    COUNT(*) as cnt
                FROM candidates
                WHERE age IS NOT NULL
                GROUP BY age_range
                ORDER BY cnt DESC
            """)
            age_dist = [(row[0], row[1]) for row in cursor.fetchall()]

            # 工作年限分布
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN work_years <= 2 THEN '0-2年'
                        WHEN work_years BETWEEN 3 AND 5 THEN '3-5年'
                        WHEN work_years BETWEEN 6 AND 10 THEN '6-10年'
                        ELSE '10年以上'
                    END as exp_range,
                    COUNT(*) as cnt
                FROM candidates
                WHERE work_years IS NOT NULL
                GROUP BY exp_range
                ORDER BY cnt DESC
            """)
            exp_dist = [(row[0], row[1]) for row in cursor.fetchall()]

            # 求职状态分布
            cursor.execute("""
                SELECT job_status, COUNT(*) as cnt 
                FROM candidates 
                WHERE job_status IS NOT NULL AND job_status != ''
                GROUP BY job_status
                ORDER BY cnt DESC
            """)
            status_dist = [(row[0], row[1]) for row in cursor.fetchall()]

            # 热门技能 Top 10
            cursor.execute("""
                SELECT skill_name, COUNT(*) as cnt 
                FROM skills 
                GROUP BY skill_name 
                ORDER BY cnt DESC 
                LIMIT 10
            """)
            top_skills = [(row[0], row[1]) for row in cursor.fetchall()]

            # 组装统计信息
            stats_parts = [
                f"- 候选人总数: {total} 人",
                f"- 学历分布: {', '.join(f'{e[0]}({e[1]}人)' for e in edu_dist)}",
                f"- 性别分布: {', '.join(f'{g[0]}({g[1]}人)' for g in gender_dist)}",
                f"- 年龄分布: {', '.join(f'{a[0]}({a[1]}人)' for a in age_dist)}",
                f"- 工作经验分布: {', '.join(f'{e[0]}({e[1]}人)' for e in exp_dist)}",
                f"- 求职状态: {', '.join(f'{s[0]}({s[1]}人)' for s in status_dist)}",
                f"- 热门技能Top10: {', '.join(f'{s[0]}({s[1]}人)' for s in top_skills)}",
            ]
            return "\n".join(stats_parts)

    except Exception as e:
        logger.warning(f"[SystemStats] 获取统计信息失败: {e}")
        return "- 系统统计信息暂时无法获取"


# ── 主处理函数 ──────────────────────────────────────────────────────────────────

# 全局查询改写器实例
_query_rewriter = QueryRewriter()
# 全局意图分类器实例
_intent_classifier = IntentClassifier()


def run_query(
    session_id: str,
    message_id: str,
    query: str,
    emp_id: str,
    history: List[Dict[str, Any]],
    pending_interaction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """无状态的请求处理入口

    调用链: http_server.py → run_query() → Harness流程

    处理流程:
    1. 从 history 恢复短期记忆
    2. 检查是否有挂起的交互需要续接
    3. 查询改写（指代消解 + 省略补全）
    4. 调用 Harness 执行查询
    5. 更新短期记忆，写回 history
    6. 返回结果

    Args:
        session_id: 会话ID
        message_id: 消息ID
        query: 用户原始查询
        emp_id: 员工工号
        history: 对话历史列表（由 SessionStore 维护）
        pending_interaction: 挂起的交互状态（如有）

    Returns:
        包含 answer, suggestions, sources, steps, history, pending_interaction 的字典
    """
    logger.info(f"[MainAgent] session={session_id} query={query!r}")

    # ── Token 用量追踪：重置计数器 ────────────────────────────────────────
    from backend.models.longcat_client import get_token_tracker
    _tracker = get_token_tracker()
    _tracker.reset()

    # ── 0. 加载双层长期记忆 ────────────────────────────────────────────────────
    from backend.memory import memory_loader as _memory_loader
    long_term_memory_context = _memory_loader.load_memory_context(emp_id)
    if long_term_memory_context:
        logger.info(f"[MainAgent] 已加载长期记忆上下文 ({len(long_term_memory_context)} chars)")

    # ── 0.5 检测记忆操作意图（保存/删除/查看）──────────────────────────────────
    memory_action = _memory_loader.process_memory_intent(emp_id, query)
    if memory_action:
        answer = _format_memory_action_response(memory_action)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        return {
            "answer": answer,
            "suggestions": ["继续提问", "查看我的记忆"],
            "sources": [],
            "steps": [],
            "history": history,
            "pending_interaction": None,
        }

    # ── 1. 恢复短期记忆 ──────────────────────────────────────────────────────
    memory_data = _extract_memory_from_history(history)
    memory = ShortTermMemory.from_dict(memory_data)
    current_turn = len(memory.entries) + 1

    # ── 1.5 意图预分类（闲聊/系统查询/招聘）─────────────────────────────────────
    # 如果有挂起交互，跳过分类直接进入续接流程
    if not pending_interaction:
        intent_type = _intent_classifier.classify(query, memory)
        logger.info(f"[MainAgent] 意图预分类: {intent_type}")

        if intent_type == "chitchat":
            result = _handle_chitchat(query, emp_id)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": result["answer"]})
            token_usage = _tracker.get()
            return {
                "answer": result["answer"],
                "suggestions": result.get("suggestions", []),
                "sources": [],
                "steps": [],
                "history": history,
                "pending_interaction": None,
                "token_usage": token_usage,
            }

        if intent_type == "system_query":
            result = _handle_system_query(query, emp_id)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": result["answer"]})
            token_usage = _tracker.get()
            return {
                "answer": result["answer"],
                "suggestions": result.get("suggestions", []),
                "sources": [],
                "steps": [],
                "history": history,
                "pending_interaction": None,
                "token_usage": token_usage,
            }

    # ── 2. 处理挂起的交互续接 ────────────────────────────────────────────────
    if pending_interaction:
        logger.info(f"[MainAgent] 续接交互: {pending_interaction.get('interaction_id')}")
        result = _handle_interaction_resume(
            query, pending_interaction, memory, session_id, emp_id
        )
        # 更新 history
        history = _update_history(history, query, result, memory, current_turn)
        return {
            "answer": result.get("answer", ""),
            "suggestions": result.get("suggestions", []),
            "sources": result.get("sources", []),
            "steps": result.get("steps", []),
            "history": history,
            "pending_interaction": result.get("pending_interaction"),
        }

    # ── 3. 查询改写 ─────────────────────────────────────────────────────────
    if _query_rewriter.needs_rewrite(query, memory):
        rewritten_query = _query_rewriter.rewrite(query, memory)
        logger.info(f"[MainAgent] 查询改写: '{query}' → '{rewritten_query}'")
    else:
        rewritten_query = query

    # ── 4. 调用 Harness 执行（注入长期记忆上下文）──────────────────────────────
    result = _execute_query(rewritten_query, session_id, emp_id, memory,
                            long_term_memory_context=long_term_memory_context)

    # ── 5. 更新短期记忆 ──────────────────────────────────────────────────────
    # 从改写后的查询中提取实体（用于后续轮次的上下文补全）
    entities = result.get("entities", {}) or _extract_entities_from_query(rewritten_query)
    entry = MemoryEntry(
        turn_id=current_turn,
        user_query=query,
        rewritten_query=rewritten_query,
        intent=result.get("intent", ""),
        entities=entities,
        result_summary=_build_result_summary(result),
        result_items=result.get("result_items", []),
    )
    memory.add_entry(entry)

    # ── 5.5 Agent 自动识别重要信息并保存到自适应记忆 ────────────────────────────
    _maybe_save_adaptive_memory(emp_id, query, rewritten_query, result)

    # ── 6. 更新 history 并返回 ───────────────────────────────────────────────
    history = _update_history(history, query, result, memory, current_turn)

    # ── 收集 Token 用量 ─────────────────────────────────────────────────
    token_usage = _tracker.get()
    logger.info(f"[MainAgent] Token用量: prompt={token_usage['prompt_tokens']}, "
                f"completion={token_usage['completion_tokens']}, "
                f"total={token_usage['total_tokens']}, calls={token_usage['llm_calls']}")

    return {
        "answer": result.get("answer", ""),
        "suggestions": result.get("suggestions", []),
        "sources": result.get("sources", []),
        "steps": result.get("steps", []),
        "history": history,
        "pending_interaction": result.get("pending_interaction"),
        "token_usage": token_usage,
    }


# ── 内部辅助函数 ────────────────────────────────────────────────────────────────

def _extract_memory_from_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从对话历史中提取短期记忆数据

    history 列表中可能包含一条 role="memory" 的特殊条目，
    用于存储序列化的短期记忆状态。
    """
    for item in history:
        if item.get("role") == "memory":
            return item.get("data", {})
    return {}


def _update_history(
    history: List[Dict[str, Any]],
    query: str,
    result: Dict[str, Any],
    memory: ShortTermMemory,
    turn_id: int,
) -> List[Dict[str, Any]]:
    """更新对话历史

    在 history 中:
    - 追加用户消息和助手回复（标准对话历史）
    - 更新 memory 条目（供下一轮恢复短期记忆）
    """
    # 追加本轮对话记录
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": result.get("answer", "")})

    # 更新或插入 memory 条目（始终保持一条 role="memory" 记录）
    memory_entry = {"role": "memory", "data": memory.to_dict()}
    # 查找并替换已有的 memory 条目
    for i, item in enumerate(history):
        if item.get("role") == "memory":
            history[i] = memory_entry
            return history
    # 不存在则追加
    history.append(memory_entry)
    return history


def _execute_query(
    query: str,
    session_id: str,
    emp_id: str,
    memory: ShortTermMemory,
    long_term_memory_context: str = "",
) -> Dict[str, Any]:
    """调用后端 Harness 执行查询，并将结果通过 LLM 转换为自然语言回复

    流程:
    1. 调用 HarnessController 获取原始结构化结果（候选人列表、匹配分数等）
    2. 将原始结果 + 用户查询 送入 LLM，生成面向用户的自然语言回答
    3. 从结果中提取候选人列表（供后续指代消解使用）
    """
    import asyncio
    from backend.harness.harness import harness as harness_controller

    context = {
        "session_id": session_id,
        "emp_id": emp_id,
        "memory_context": memory.get_recent_context(),
        "active_entities": memory.active_entities,
        "long_term_memory": long_term_memory_context,
    }

    try:
        # 在同步上下文中运行异步 harness
        loop = asyncio.new_event_loop()
        try:
            harness_result = loop.run_until_complete(
                harness_controller.execute(query, context)
            )
        finally:
            loop.close()

        if harness_result.get("success"):
            final = harness_result.get("result", {})
            # 将原始结构化结果转换为自然语言回复（注入长期记忆供 LLM 参考）
            answer = _generate_natural_language_answer(
                query, final, harness_result,
                memory_context=long_term_memory_context
            )
            # 提取候选人列表用于后续指代消解
            result_items = _extract_result_items(final)
            # 生成追问建议
            suggestions = _generate_suggestions(query, final)
            return {
                "answer": answer,
                "suggestions": suggestions,
                "sources": final.get("sources", []),
                "steps": harness_result.get("steps", []),
                "intent": final.get("intent", ""),
                "entities": final.get("entities", {}),
                "result_items": result_items,
            }
        else:
            error_msg = harness_result.get("error", "处理失败")
            return {"answer": f"⚠️ {error_msg}", "suggestions": ["换个说法试试"], "sources": []}

    except Exception as e:
        logger.error(f"[MainAgent] Harness 执行异常: {e}", exc_info=True)
        return {
            "answer": f"⚠️ 处理异常: {str(e)}，请稍后重试。",
            "suggestions": ["重新提问", "换个说法试试"],
            "sources": [],
        }


def _generate_natural_language_answer(
    query: str, raw_result: Dict[str, Any], harness_meta: Dict[str, Any],
    memory_context: str = "",
) -> str:
    """将 Harness 的结构化结果转换为面向用户的自然语言回复

    使用 LLM 对原始数据进行总结，生成可读性强的回复。
    如果 LLM 调用失败，则使用规则式回退。
    """
    from backend.models.longcat_client import chat_completion
    from backend.utils.candidate_category import build_category_knowledge

    # 准备结果摘要（防止 token 过长）
    result_summary = _summarize_raw_result(raw_result)

    # 获取动态的候选人类别定义知识
    category_knowledge = build_category_knowledge()

    # 构建记忆上下文段落（如有）
    memory_section = ""
    if memory_context:
        memory_section = f"\n{memory_context}\n"

    system_prompt = f"""你是一个智能招聘匹配系统的回复助手。你的任务是将系统的检索和匹配结果转化为专业、友好的自然语言回复。
{memory_section}
{category_knowledge}


规则：
1. 如果结果中包含候选人列表，必须逐一列出所有候选人（姓名、匹配度、核心优势），不得省略任何一位，不得用"其余候选人"等方式概括
2. 使用Markdown格式，便于前端渲染
3. 保持简洁专业，重点突出匹配度和关键信息
4. 如果检索结果为空（候选人列表为空），才告知未找到并建议调整条件
5. 不要暴露系统内部的技术细节（如subtask_id、score的原始值等）
6. 匹配分数用百分比表示（如 0.85 → 85%）
7. **重要**：排序时优先展示满足年限要求的候选人，不满足年限要求的放在后面并注明
8. **重要**：如果结果中包含SHAP贡献信息，必须在每位候选人下方展示其SHAP关键贡献特征（如"主要匹配维度：skill_match(+0.05), experience_match(+0.03)"），这是系统可解释性的核心功能
9. 在结果末尾添加一个"SHAP可解释性说明"小节，简要解释SHAP值的含义（正值=正向贡献，负值=不利因素）
10. **重要**：如果候选人数据中包含"教育经历"字段（多段学历列表），必须完整展示其教育路径。格式示例："专科·深圳职业技术大学(计算机) → 本科·杭州电子科技大学(软件工程)[非全日制] → 硕士·北京邮电大学(计算机科学)[211]"。注意标注院校层次（985/211/双一流）和全日制/非全日制属性，这是系统的核心特色功能
11. **极其重要**：你只负责格式化和呈现系统已经检索到的结果，不要自行判断候选人是否满足用户的搜索条件。即使你认为某个候选人不完全符合用户的查询意图，也必须如实展示系统返回的所有候选人信息。系统已经完成了筛选，你的工作是展示结果而非二次过滤。如果用户提到了特定院校（如"上交"、"川大"），你应该检查每位候选人的"教育经历"字段是否包含该院校，将匹配的候选人优先展示并明确标注，其余候选人作为补充推荐展示
12. **重要**：常见院校别名映射——"上交"="上海交通大学"、"电子科大"="电子科技大学"、"川大"="四川大学"、"北邮"="北京邮电大学"、"哈工大"="哈尔滨工业大学"、"华科"="华中科技大学"、"浙大"="浙江大学"、"北大"="北京大学"、"清华"="清华大学"、"复旦"="复旦大学"、"南大"="南京大学"、"中科大"="中国科学技术大学"、"西电"="西安电子科技大学"、"成电"="电子科技大学"、"武大"="武汉大学"。回复中使用完整校名"""

    user_prompt = f"""用户查询：{query}

系统返回的原始结果：
{result_summary}

请将以上结果转换为面向用户的自然语言回复。"""

    try:
        response = chat_completion(system=system_prompt, user=user_prompt, temperature=0.3)
        answer = (response.content or "").strip()
        if answer:
            return answer
    except Exception as e:
        logger.warning(f"[MainAgent] LLM 结果总结失败: {e}, 使用规则式回退")

    # 规则式回退：直接从结构化数据生成简单回复
    return _fallback_format_result(query, raw_result)


def _summarize_raw_result(raw_result: Dict[str, Any]) -> str:
    """将原始结果压缩为 LLM 可处理的摘要（控制 token 长度）"""
    import json

    parts = []

    # 候选人列表 - 兼容多种结果结构
    candidates = _extract_candidates_list(raw_result)

    if not candidates:
        # 经过匹配评估后无合格候选人 — 提取约束信息帮助 LLM 生成友好回复
        constraints_info = {}
        sources = [raw_result]
        if "results" in raw_result and isinstance(raw_result["results"], list):
            sources = raw_result["results"]
        for src in sources:
            if isinstance(src, dict):
                if src.get("constraints_detected"):
                    constraints_info = src["constraints_detected"]
                matched = src.get("matched")
                if isinstance(matched, dict):
                    constraints_info["total_evaluated"] = matched.get("total_evaluated", 0)
        parts.append("经过严格筛选，未找到完全符合条件的候选人。")
        if constraints_info:
            parts.append(f"已应用的筛选条件: {json.dumps(constraints_info, ensure_ascii=False)}")
        parts.append("建议：放宽部分条件（如院校范围、毕业年份）后重试。")
        return "\n".join(parts)

    if candidates and isinstance(candidates, list):
        # 只取前10个候选人的关键信息
        top_candidates = candidates[:10]
        simplified = []

        # 获取SHAP解释（如果有）— 兼容多子任务聚合结构
        shap_map = {}
        shap_sources = [raw_result]
        if "results" in raw_result and isinstance(raw_result["results"], list):
            shap_sources = raw_result["results"]
        for src in shap_sources:
            if isinstance(src, dict):
                for s in src.get("shap_explanations", []):
                    shap_map[s.get("candidate_id")] = s.get("shap_values", {})

        for i, c in enumerate(top_candidates):
            data = c.get("data", c)
            cand_id = data.get("id") or c.get("candidate_id")
            info = {
                "排名": i + 1,
                "姓名": data.get("name", f"候选人{i+1}"),
                "匹配分": round(c.get("score", 0), 3) if "score" in c else None,
                "最高学历": data.get("highest_education", ""),
                "工作年限": data.get("work_years", ""),
                "技能": data.get("skills", [])[:5] if isinstance(data.get("skills"), list) else "",
                "满足年限要求": c.get("meets_experience_req", True),
                "目标院校匹配": c.get("school_match", False) if c.get("school_match") is not None else None,
                "毕业年份匹配": c.get("grad_year_match", True) if c.get("grad_year_match") is not None else None,
            }
            # 完整教育经历路径（多段学历）
            edu_history = data.get("education_history", [])
            if edu_history:
                edu_path = []
                for edu in edu_history:
                    seg = f"{edu.get('degree', '')}·{edu.get('school', '')}({edu.get('major', '')})"
                    if edu.get('school_tier') and edu.get('school_tier') != '普通本科':
                        seg += f"[{edu.get('school_tier')}]"
                    if edu.get('is_fulltime') == 0:
                        seg += "[非全日制]"
                    period = f"{edu.get('start_date', '')[:7]}~{edu.get('end_date', '')[:7]}"
                    seg += f" {period}"
                    edu_path.append(seg)
                info["教育经历"] = edu_path
            else:
                # 回退：使用单字段
                info["学校"] = data.get("school", "")
                info["专业"] = data.get("major", "")
            # 附加SHAP解释（前3个主要贡献特征）
            shap_vals = shap_map.get(cand_id, {})
            if shap_vals:
                sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                info["SHAP主要贡献"] = [{"特征": k, "贡献值": round(v, 4)} for k, v in sorted_shap]
            # 过滤空值
            info = {k: v for k, v in info.items() if v or v == 0 or v is False}
            simplified.append(info)
        parts.append(f"匹配到 {len(candidates)} 位候选人，Top {len(simplified)} 如下：")
        parts.append(json.dumps(simplified, ensure_ascii=False, indent=2))
    elif raw_result.get("data"):
        # 其他类型的结果
        data = raw_result["data"]
        if isinstance(data, dict):
            parts.append(json.dumps(data, ensure_ascii=False, indent=2)[:1500])
        else:
            parts.append(str(data)[:1500])
    elif raw_result.get("explanation"):
        parts.append(f"可解释性分析结果: {json.dumps(raw_result['explanation'], ensure_ascii=False)[:1500]}")
    else:
        parts.append(f"结果: {json.dumps(raw_result, ensure_ascii=False)[:1500]}")

    return "\n".join(parts)


def _fallback_format_result(query: str, raw_result: Dict[str, Any]) -> str:
    """规则式回退格式化（LLM 不可用时）"""
    candidates = _extract_candidates_list(raw_result)

    if candidates and isinstance(candidates, list):
        lines = [f"根据您的查询「{query}」，为您找到以下候选人：\n"]
        for i, c in enumerate(candidates[:10]):
            data = c.get("data", c)
            name = data.get("name", f"候选人{i+1}")
            score = c.get("score", 0)
            lines.append(f"**{i+1}. {name}** - 匹配度 {score*100:.0f}%")
            # 优先展示完整教育经历路径
            edu_history = data.get("education_history", [])
            if edu_history:
                edu_parts = []
                for edu in edu_history:
                    seg = f"{edu.get('degree', '')}·{edu.get('school', '')}"
                    if edu.get('school_tier') and edu.get('school_tier') != '普通本科':
                        seg += f"({edu.get('school_tier')})"
                    if edu.get('is_fulltime') == 0:
                        seg += "(非全日制)"
                    edu_parts.append(seg)
                lines.append(f"   教育经历: {'→'.join(edu_parts)}")
            else:
                edu = data.get("highest_education", "")
                if edu:
                    lines.append(f"   学历: {edu}")
        lines.append(f"\n共找到 {len(candidates)} 位匹配候选人，以上展示前 {min(10, len(candidates))} 位。")
        return "\n".join(lines)
    else:
        return f"已处理您的查询「{query}」，但暂未找到匹配的候选人。您可以尝试调整搜索条件。"


def _extract_candidates_list(raw_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从原始结果中提取候选人列表（兼容多种结果结构）

    处理以下几种情况:
    - 多子任务聚合结果: {"results": [subtask1_result, subtask2_result, ...]}
    - RAG检索结果: {"candidates": [{candidate_id, score, data}]}
    - 匹配评估结果: {"matched": {"matched_candidates": [{candidate_id, candidate, match_score}]}}
    - 直接列表: [candidate1, candidate2, ...]

    重要原则: 如果 matched 字段存在（来自 MatchingEvaluationSkill），以 matched 为准，
    即使为空也不回退到 RAG 原始结果。空结果表示经过严格筛选后确实无合格候选人。
    """
    # 处理多子任务聚合结果 — 从所有子任务中找到包含候选人列表的最终结果
    if "results" in raw_result and isinstance(raw_result["results"], list):
        # 逆序查找：后面的子任务通常是最终的匹配/排序结果
        for sub_result in reversed(raw_result["results"]):
            if isinstance(sub_result, dict):
                sub_candidates = _extract_candidates_list(sub_result)
                if sub_candidates:
                    return sub_candidates
        # 再次逆序查找：如果所有子任务的 matched 都为空，检查是否有 matched 字段存在
        # 如果有 matched 字段说明评估已执行，空结果是正确的（不应回退到 RAG）
        for sub_result in reversed(raw_result["results"]):
            if isinstance(sub_result, dict) and "matched" in sub_result:
                return []  # matched 存在但为空 → 确实无结果
        return []

    # 检查 matched -> matched_candidates (来自 MatchingEvaluationSkill)
    # 重要：matched 字段存在即为权威结果，即使 matched_candidates 为空也以此为准
    matched = raw_result.get("matched")
    if isinstance(matched, dict):
        matched_candidates = matched.get("matched_candidates", [])
        if matched_candidates:
            # 统一格式: 将 match_score 映射为 score, candidate 映射为 data
            normalized = []
            for mc in matched_candidates:
                normalized.append({
                    "candidate_id": mc.get("candidate_id"),
                    "score": mc.get("match_score", 0),
                    "data": mc.get("candidate", mc),
                })
            return normalized
        else:
            # matched 存在但 candidates 为空 → 经评估后确实无合格候选人
            return []

    # 检查 candidates (dict with inner candidates key, 来自 RAGRetrievalSkill)
    candidates = raw_result.get("candidates")
    if isinstance(candidates, dict):
        return candidates.get("candidates", [])
    if isinstance(candidates, list):
        return candidates

    # 检查 matched 为 list 的情况
    if isinstance(matched, list):
        return matched

    return []


def _extract_result_items(raw_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从原始结果中提取候选人列表（供短期记忆的指代消解使用）"""
    candidates = _extract_candidates_list(raw_result)

    if not candidates:
        return []

    items = []
    for c in candidates[:10]:
        data = c.get("data", c)
        items.append({
            "name": data.get("name", ""),
            "id": data.get("id") or c.get("candidate_id", ""),
            "score": c.get("score", c.get("match_score", 0)),
        })
    return items


def _generate_suggestions(query: str, raw_result: Dict[str, Any]) -> List[str]:
    """根据查询和结果生成追问建议"""
    suggestions = []
    candidates = _extract_candidates_list(raw_result)

    if candidates and isinstance(candidates, list):
        suggestions.append("查看第一位候选人的详细信息")
        if len(candidates) > 5:
            suggestions.append("只看前3位匹配度最高的")
        suggestions.append("换一批候选人看看")
    else:
        suggestions.append("帮我找Java高级工程师")
        suggestions.append("推荐有5年经验的产品经理")

    return suggestions[:3]


def _handle_interaction_resume(
    user_answer: str,
    pending: Dict[str, Any],
    memory: ShortTermMemory,
    session_id: str,
    emp_id: str,
) -> Dict[str, Any]:
    """处理挂起交互的续接

    当 Agent 需要用户确认/选择时，会将执行上下文序列化到 pending_interaction。
    用户回答后，从 pending 恢复上下文，跳过意图解析直接执行。
    """
    resume_context = pending.get("resume_context", {})
    resolved_query = resume_context.get("resolved_query", user_answer)

    # 将用户选择融入查询
    interaction_type = pending.get("interaction_type", "input")
    if interaction_type == "confirm":
        if user_answer.lower() in ("是", "yes", "确认", "y", "对"):
            # 用户确认，执行原计划
            return _execute_query(resolved_query, session_id, emp_id, memory)
        else:
            return {"answer": "好的，已取消操作。有什么其他问题可以帮您？", "suggestions": []}
    elif interaction_type == "select":
        # 用户选择了某个选项，融入查询
        enhanced_query = f"{resolved_query}，用户选择: {user_answer}"
        return _execute_query(enhanced_query, session_id, emp_id, memory)
    else:
        # 自由输入，补充到查询中
        enhanced_query = f"{resolved_query}，补充信息: {user_answer}"
        return _execute_query(enhanced_query, session_id, emp_id, memory)


def _build_result_summary(result: Dict[str, Any]) -> str:
    """构建结果摘要（用于存入短期记忆）"""
    answer = result.get("answer", "")
    if len(answer) > 100:
        return answer[:100] + "..."
    return answer


def _extract_entities_from_query(query: str) -> Dict[str, Any]:
    """从查询中提取关键实体（用于短期记忆的活跃实体追踪）

    使用规则匹配提取常见的招聘相关实体：职位类型、经验年限、技能等。
    这些实体在后续轮次中用于上下文补全。
    """
    import re

    entities: Dict[str, Any] = {}

    # 提取职位类型
    position_patterns = [
        (r"(前端|后端|全栈|移动端|iOS|Android|测试|运维|架构|算法|数据|AI|机器学习|深度学习|NLP|CV)",
         "position_type"),
    ]
    for pattern, key in position_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            entities[key] = match.group(1)

    # 提取职位级别
    level_patterns = [
        (r"(初级|中级|高级|资深|专家|leader|主管|经理|总监)", "position_level"),
    ]
    for pattern, key in level_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            entities[key] = match.group(1)

    # 提取经验年限（支持中文数字和阿拉伯数字）
    exp_match = re.search(r"(\d+)\s*年", query)
    if exp_match:
        entities["experience_years"] = exp_match.group(1) + "年"
    else:
        cn_num_map = {"一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
                      "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
        cn_exp_match = re.search(r"([一二两三四五六七八九十]+)\s*年", query)
        if cn_exp_match:
            cn_num = cn_exp_match.group(1)
            arabic = cn_num_map.get(cn_num, cn_num)
            entities["experience_years"] = arabic + "年"

    # 提取职位名称（如"工程师"、"产品经理"等）
    role_match = re.search(r"(工程师|开发[者人员]*|程序员|设计师|产品经理|分析师|架构师|运营)", query)
    if role_match:
        entities["role"] = role_match.group(1)

    # 提取技术栈关键词
    tech_patterns = r"(Java|Python|Go|Golang|C\+\+|JavaScript|TypeScript|React|Vue|Spring|MySQL|Redis|Kafka|Docker|K8s|Kubernetes)"
    tech_matches = re.findall(tech_patterns, query, re.IGNORECASE)
    if tech_matches:
        entities["tech_stack"] = ", ".join(tech_matches)

    return entities


# ── 长期记忆相关辅助函数 ──────────────────────────────────────────────────────────

def _format_memory_action_response(action: Dict[str, Any]) -> str:
    """将记忆操作结果格式化为用户可读的回复"""
    action_type = action.get("action")

    if action_type == "save_persistent":
        content = action.get("content", "")
        category_map = {"rule": "规则", "preference": "偏好", "definition": "定义"}
        cat = category_map.get(action.get("category", ""), "记忆")
        return f"✅ 已记住！我将这条{cat}保存到了长期记忆中：\n\n> {content}\n\n以后的对话中我会遵循这条{cat}。"

    elif action_type == "delete":
        keyword = action.get("keyword", "")
        count = action.get("count", 0)
        if count > 0:
            return f"✅ 已删除与「{keyword}」相关的 {count} 条记忆。"
        else:
            return f"未找到与「{keyword}」相关的记忆，无需删除。"

    elif action_type == "view":
        persistent = action.get("persistent", [])
        adaptive = action.get("adaptive", [])

        parts = []
        if persistent:
            parts.append("**📌 长期记忆（你明确要求记住的）：**")
            for i, mem in enumerate(persistent, 1):
                cat_map = {"rule": "规则", "preference": "偏好", "definition": "定义"}
                cat = cat_map.get(mem.get("category", ""), "其他")
                parts.append(f"  {i}. [{cat}] {mem['content']}")
        else:
            parts.append("📌 长期记忆：暂无")

        if adaptive:
            parts.append("\n**🧠 系统观察（自动识别的习惯）：**")
            for i, mem in enumerate(adaptive, 1):
                score = mem.get("importance", 0)
                parts.append(f"  {i}. {mem['content']} (重要性: {score:.0%})")
        else:
            parts.append("\n🧠 系统观察：暂无")

        return "\n".join(parts)

    return "记忆操作完成。"


def _maybe_save_adaptive_memory(
    user_id: str, query: str, rewritten_query: str, result: Dict[str, Any]
) -> None:
    """Agent 自动判断是否需要保存记忆

    通过 memory_loader.save_memory() 统一入口保存，由重要性评分自动决定分层：
    - importance >= 0.9 → Layer 1（永久记忆）
    - importance < 0.9  → Layer 2（自适应记忆，会衰减）

    触发条件（满足任一即保存）：
    1. 用户查询中包含明确的筛选偏好（如"只看985"、"不要专科"）
    2. 用户对结果有明确的正/负反馈
    3. 用户连续多次搜索相似类型的岗位（模式识别）
    """
    from backend.memory import memory_loader as _memory_loader

    # 检测筛选偏好表达
    preference_patterns = [
        (r"只[看要找](.{2,15})", "用户偏好只看{match}类型的候选人"),
        (r"不[要看](.{2,15})", "用户不想看{match}类型的候选人"),
        (r"优先(.{2,15})", "用户优先关注{match}"),
        (r"排除(.{2,15})", "用户要求排除{match}"),
    ]

    import re
    for pattern, template in preference_patterns:
        match = re.search(pattern, query)
        if match:
            content = template.format(match=match.group(1))
            # 统一通过 save_memory 入口，由重要性评分自动决定分层
            result_info = _memory_loader.save_memory(
                user_id=user_id,
                content=content,
                query_context=query,
                category="pattern",
            )
            logger.info(
                f"[MainAgent] 记忆已保存: layer={result_info['layer']}, "
                f"importance={result_info['importance']:.2f}, content={content}"
            )
            break
