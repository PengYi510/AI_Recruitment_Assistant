"""记忆加载器 - 统一由 Agent 评估重要性决定分层

设计原则：
- 不再依赖"用户是否说了记住"来决定分层
- 统一由 Agent 评估每条信息的重要性分数 (0~1)
- importance >= 0.9 → Layer 1 (永久记忆，不衰减)
- importance < 0.9  → Layer 2 (自适应记忆，会衰减)
- 用户显式说"记住/永远/始终"只是重要性评估的加分因素之一，不是唯一决定因素
- 查看/删除记忆仍保留触发词检测（这是用户管理操作，不是保存操作）

职责：
1. 加载用户的永久记忆（Layer 1）和自适应记忆（Layer 2）
2. 合并为统一的记忆上下文文本，供 system prompt 注入
3. 提供统一的记忆写入入口：评估重要性 → 自动分层
4. 提供查看/删除记忆的管理操作
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, List, Optional

from backend.memory.persistent_memory import PersistentMemoryStore, persistent_memory
from backend.memory.adaptive_memory import AdaptiveMemoryStore, adaptive_memory

logger = logging.getLogger(__name__)

# 重要性阈值：>= 此值进入 Layer 1（永久记忆）
PERMANENT_THRESHOLD = 0.9

# 查看记忆的触发词（管理操作，保留）
VIEW_TRIGGERS = [
    "你记住了什么", "你记得什么", "看看记忆", "查看记忆",
    "我的偏好", "我的规则", "我定义了什么",
    "what do you remember", "show memory", "my preferences",
]

# 删除记忆的触发词（管理操作，保留）
DELETE_TRIGGERS = [
    "忘掉", "忘记", "取消", "不再", "别再", "删除这条",
    "去掉", "移除", "撤销", "作废", "不要记",
    "forget", "remove", "cancel", "delete",
]


class MemoryLoader:
    """双层记忆加载与管理器（重要性驱动分层）"""

    def __init__(
        self,
        persistent_store: PersistentMemoryStore = None,
        adaptive_store: AdaptiveMemoryStore = None,
    ):
        self.persistent = persistent_store or persistent_memory
        self.adaptive = adaptive_store or adaptive_memory

    # ── 加载记忆（注入 prompt）────────────────────────────────────────────────

    def load_memory_context(self, user_id: str) -> str:
        """加载用户的完整记忆上下文

        合并两层记忆为一段文本，用于注入到 system prompt 中。
        如果两层都为空，返回空字符串。
        """
        parts = []

        # Layer 1: 永久记忆（优先级最高，放在前面）
        persistent_text = self.persistent.format_for_prompt(user_id)
        if persistent_text:
            parts.append(persistent_text)

        # Layer 2: 自适应记忆（补充信息）
        adaptive_text = self.adaptive.format_for_prompt(user_id, top_n=10)
        if adaptive_text:
            parts.append(adaptive_text)

        if not parts:
            return ""

        header = "═══ 用户记忆上下文（请在回复中遵循以下用户偏好和规则）═══"
        footer = "═══ 记忆上下文结束 ═══"
        return f"\n{header}\n" + "\n\n".join(parts) + f"\n{footer}\n"

    # ── 记忆管理操作（查看/删除）──────────────────────────────────────────────

    def process_memory_intent(self, user_id: str, query: str) -> Optional[Dict[str, Any]]:
        """处理对话中的记忆管理意图（仅查看和删除）

        注意：保存操作不再在这里处理，改为由 save_memory() 统一入口处理。

        Args:
            user_id: 用户标识
            query: 用户原始输入

        Returns:
            操作结果字典，无记忆管理操作时返回 None
        """
        # 检测查看意图
        if any(trigger in query for trigger in VIEW_TRIGGERS):
            memories = self.persistent.get_all(user_id)
            adaptive_mems = self.adaptive.get_active(user_id, limit=10)
            return {
                "action": "view",
                "persistent": memories,
                "adaptive": adaptive_mems,
            }

        # 检测删除意图
        if any(trigger in query for trigger in DELETE_TRIGGERS):
            keyword = self._extract_delete_target(query)
            if keyword:
                # 同时在两层中搜索并删除
                count_p = self.persistent.deactivate_by_keyword(user_id, keyword)
                count_a = self._deactivate_adaptive_by_keyword(user_id, keyword)
                return {
                    "action": "delete",
                    "keyword": keyword,
                    "count": count_p + count_a,
                    "detail": {"persistent": count_p, "adaptive": count_a},
                }

        return None

    # ── 统一记忆写入入口（重要性驱动分层）────────────────────────────────────

    def save_memory(self, user_id: str, content: str, query_context: str = "",
                    category: str = "observation") -> Dict[str, Any]:
        """统一的记忆保存入口 - 由 Agent 评估重要性自动决定分层

        流程：
        1. 提取核心记忆内容（去除触发词前缀）
        2. 评估重要性分数（综合内容特征 + 用户语气 + 上下文）
        3. importance >= 0.9 → Layer 1（永久记忆，不衰减）
        4. importance < 0.9  → Layer 2（自适应记忆，会衰减）

        Args:
            user_id: 用户标识
            content: 要保存的记忆内容
            query_context: 原始用户输入（用于评估语气强度）
            category: 分类 (rule/preference/definition/observation/pattern/insight)

        Returns:
            {"layer": "persistent"|"adaptive", "memory_id": int, "importance": float}
        """
        # 评估重要性
        importance = self.assess_importance(content, query_context)

        # 自动分类
        if category == "observation":
            category = self._classify_category(content)

        keywords = self._extract_keywords(content)

        if importance >= PERMANENT_THRESHOLD:
            # Layer 1: 永久记忆
            memory_id = self.persistent.save(
                user_id=user_id,
                content=content,
                category=category,
                keywords=keywords,
            )
            logger.info(
                f"[MemoryLoader] → Layer 1 (永久): importance={importance:.2f}, "
                f"content={content[:50]}..."
            )
            return {"layer": "persistent", "memory_id": memory_id, "importance": importance}
        else:
            # Layer 2: 自适应记忆
            memory_id = self.adaptive.save(
                user_id=user_id,
                content=content,
                category=category,
                importance=importance,
            )
            logger.info(
                f"[MemoryLoader] → Layer 2 (自适应): importance={importance:.2f}, "
                f"content={content[:50]}..."
            )
            return {"layer": "adaptive", "memory_id": memory_id, "importance": importance}

    # ── 重要性评估（核心逻辑）────────────────────────────────────────────────

    def assess_importance(self, content: str, query_context: str = "") -> float:
        """评估一条信息的重要性分数 (0~1)

        评分维度：
        1. 内容持久性 (0~0.3): 这条信息是否具有长期有效性？
           - 包含"等于/对应/映射"等定义性表达 → +0.25
           - 包含"永远/始终/所有"等全称量词 → +0.2
           - 包含具体数值阈值 → +0.15

        2. 用户意图强度 (0~0.4): 用户表达的坚定程度
           - 使用"记住/永远/始终/以后都"等强指令 → +0.4
           - 使用"默认/习惯/偏好"等中等指令 → +0.2
           - 使用"试试/暂时/这次"等弱化词 → -0.15

        3. 信息独特性 (0~0.2): 是否是具体的、可操作的规则
           - 包含明确的条件→结论结构 → +0.15
           - 包含专有名词/术语 → +0.1

        4. 基础分 (0.3): 所有被识别为值得记忆的信息的起始分

        Returns:
            重要性分数 (0~1)
        """
        score = 0.3  # 基础分

        # ── 维度1: 内容持久性 ──
        # 定义性表达（"A等于B"、"A对应B"）
        if re.search(r'(等于|等同于|约等于|相当于|对应|映射|就是|代表|意味着)', content):
            score += 0.25
        # 全称量词（"永远"、"所有"、"任何时候"）
        elif re.search(r'(永远|始终|所有|任何|一律|统一|全部)', content):
            score += 0.2
        # 具体数值阈值
        elif re.search(r'\d+[年万kK%名]', content):
            score += 0.15

        # ── 维度2: 用户意图强度（从原始 query 中判断）──
        context = query_context or content
        # 强指令
        strong_signals = ["记住", "永远", "始终", "以后都", "以后默认", "一直",
                          "必须", "绝对", "规定", "定义", "remember", "always"]
        if any(s in context for s in strong_signals):
            score += 0.4
        # 中等指令
        elif any(s in context for s in ["默认", "习惯", "偏好", "一般", "通常", "prefer"]):
            score += 0.2
        # 弱化词（降分）
        if any(s in context for s in ["试试", "暂时", "这次", "先", "临时", "可能"]):
            score -= 0.15

        # ── 维度3: 信息独特性 ──
        # 条件→结论结构
        if re.search(r'(如果|当|只要).+(就|则|那么|等于|算)', content):
            score += 0.15
        # 包含专有名词（大写英文、特定术语）
        elif re.search(r'[A-Z]{2,}|QS|985|211|Top\s*\d+', content):
            score += 0.1

        return max(0.0, min(1.0, score))

    # ── 定期维护 ──────────────────────────────────────────────────────────────

    def run_consolidation(self, user_id: str) -> Dict[str, int]:
        """触发自适应记忆的定期总结与清理

        同时检查 Layer 2 中是否有记忆因被频繁引用而升级到 Layer 1 的条件。
        """
        stats = self.adaptive.consolidate(user_id)

        # 检查是否有 Layer 2 记忆需要升级到 Layer 1
        promoted = self._promote_high_importance_memories(user_id)
        stats["promoted_to_persistent"] = promoted

        return stats

    def _promote_high_importance_memories(self, user_id: str) -> int:
        """将 Layer 2 中重要性达到阈值的记忆升级到 Layer 1"""
        high_mems = self.adaptive.get_active(user_id, min_importance=PERMANENT_THRESHOLD, limit=10)
        promoted = 0
        for mem in high_mems:
            # 转移到 Layer 1
            category = mem.get("category", "observation")
            content = mem["content"]
            keywords = self._extract_keywords(content)
            self.persistent.save(user_id, content, category, keywords)
            # 从 Layer 2 停用
            self.adaptive.reference(mem["id"])  # 标记引用
            # 注意：不直接停用，让衰减机制自然处理
            promoted += 1
            logger.info(f"[MemoryLoader] 升级记忆到 Layer 1: {content[:50]}...")
        return promoted

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_delete_target(query: str) -> str:
        """从删除指令中提取目标关键词"""
        prefixes = [
            r"^(请你?)?忘掉[，,：:\s]*",
            r"^(请你?)?忘记[，,：:\s]*",
            r"^取消[，,：:\s]*",
            r"^删除[，,：:\s]*",
            r"^去掉[，,：:\s]*",
            r"^不再[，,：:\s]*",
        ]
        content = query.strip()
        for prefix in prefixes:
            content = re.sub(prefix, "", content, flags=re.IGNORECASE)

        suffixes = [r"的?(规则|定义|偏好|设置|记忆)$", r"这条$"]
        for suffix in suffixes:
            content = re.sub(suffix, "", content)

        return content.strip()

    def _deactivate_adaptive_by_keyword(self, user_id: str, keyword: str) -> int:
        """在自适应记忆中按关键词停用"""
        mems = self.adaptive.get_active(user_id, min_importance=0.0, limit=50)
        count = 0
        for mem in mems:
            if keyword in mem.get("content", ""):
                # 通过设置极低重要性让衰减清理掉
                with self.adaptive._get_conn() as conn:
                    conn.execute(
                        "UPDATE adaptive_memory SET is_active = 0 WHERE id = ?",
                        (mem["id"],)
                    )
                count += 1
        return count

    @staticmethod
    def _classify_category(content: str) -> str:
        """根据内容自动分类"""
        if re.search(r'(等于|等同于|约等于|相当于|对应|映射|就是)', content):
            return "rule"
        elif re.search(r'(定义|是指|意思是|代表|含义)', content):
            return "definition"
        elif re.search(r'(偏好|喜欢|优先|默认|习惯|倾向)', content):
            return "preference"
        elif re.search(r'(经常|总是|频繁|每次|一直)', content):
            return "pattern"
        return "observation"

    @staticmethod
    def _extract_keywords(content: str) -> str:
        """从内容中提取关键词（逗号分隔）"""
        words = re.findall(r'[A-Za-z0-9]+|[\u4e00-\u9fff]{2,4}', content)
        seen = set()
        keywords = []
        for w in words:
            if w.lower() not in seen and len(w) >= 2:
                seen.add(w.lower())
                keywords.append(w)
                if len(keywords) >= 5:
                    break
        return ",".join(keywords)

    @staticmethod
    def extract_memory_content(query: str) -> str:
        """从用户输入中提取核心记忆内容（去除触发词前缀）

        公开方法，供 main_agent 调用。
        例如："记住，QS前50等于985" → "QS前50等于985"
        """
        prefixes = [
            r"^(请你?)?记住[，,：:\s]*",
            r"^(请你?)?帮我记[住下][，,：:\s]*",
            r"^(你要)?记得[，,：:\s]*",
            r"^别忘了[，,：:\s]*",
            r"^以后(都|默认|搜索)?[，,：:\s]*",
            r"^从现在开始[，,：:\s]*",
            r"^永远[，,：:\s]*",
            r"^始终[，,：:\s]*",
        ]
        content = query.strip()
        for prefix in prefixes:
            content = re.sub(prefix, "", content, flags=re.IGNORECASE)
        return content.strip()


# 全局单例
memory_loader = MemoryLoader()
