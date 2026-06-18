"""Layer 2: 智能自适应记忆 (Adaptive Memory)

Agent 自动识别并保存的重要信息，具有时间衰减和重要性评分机制。
特点：
- Agent 主动识别对话中的重要信息并自动保存
- 每条记忆有重要性分数 (0~1)，随时间衰减
- 定期触发总结/清理：合并相似记忆、删除低分记忆
- 重要性分数受以下因素影响：
  - 初始重要性（由 Agent 评估）
  - 时间衰减（半衰期可配置，默认7天）
  - 被引用次数（每次被使用时加分）
  - 用户反馈（正面反馈加分，负面减分）

数据表结构：
    adaptive_memory (
        id              INTEGER PRIMARY KEY,
        user_id         TEXT NOT NULL,
        content         TEXT NOT NULL,       -- 记忆内容
        category        TEXT DEFAULT 'observation',  -- observation/pattern/insight
        importance      REAL DEFAULT 0.5,    -- 当前重要性分数 (0~1)
        initial_importance REAL DEFAULT 0.5, -- 初始重要性（不衰减的基准）
        reference_count INTEGER DEFAULT 0,   -- 被引用次数
        last_referenced TIMESTAMP,           -- 最后被引用时间
        created_at      TIMESTAMP,
        updated_at      TIMESTAMP,
        decay_rate      REAL DEFAULT 0.1,    -- 衰减速率（每天）
        is_active       INTEGER DEFAULT 1
    )
"""

from __future__ import annotations

import math
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from datetime import datetime, timedelta

from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

_MEMORY_DB_PATH = str(DATA_DIR / "memory.db")

# ── 配置 ──────────────────────────────────────────────────────────────────────
DECAY_HALF_LIFE_DAYS = 7.0       # 半衰期：7天后重要性降为初始值的一半
MIN_IMPORTANCE = 0.05            # 低于此分数的记忆将在清理时被删除
MAX_ADAPTIVE_MEMORIES = 50       # 每用户最多保留的自适应记忆条数
REFERENCE_BOOST = 0.1            # 每次被引用时的加分
CONSOLIDATION_SIMILARITY = 0.8   # 合并相似度阈值（预留，暂用关键词匹配）


class AdaptiveMemoryStore:
    """智能自适应记忆存储"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or _MEMORY_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """初始化自适应记忆表"""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS adaptive_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'observation',
                    importance REAL DEFAULT 0.5,
                    initial_importance REAL DEFAULT 0.5,
                    reference_count INTEGER DEFAULT 0,
                    last_referenced TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    decay_rate REAL DEFAULT 0.1,
                    is_active INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_am_user_active_importance
                    ON adaptive_memory(user_id, is_active, importance DESC);
            """)

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def save(self, user_id: str, content: str, category: str = "observation",
             importance: float = 0.5) -> int:
        """保存一条自适应记忆

        Args:
            user_id: 用户标识
            content: 记忆内容
            category: 分类 (observation/pattern/insight)
            importance: 初始重要性 (0~1)

        Returns:
            新记录的 ID
        """
        importance = max(0.0, min(1.0, importance))
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO adaptive_memory
                   (user_id, content, category, importance, initial_importance)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, content, category, importance, importance)
            )
            record_id = cursor.lastrowid
            logger.info(
                f"[AdaptiveMemory] 保存记忆 #{record_id}: "
                f"[{category}|{importance:.2f}] {content[:50]}..."
            )

            # 检查是否超出上限，超出则清理低分记忆
            self._enforce_limit(conn, user_id)
            return record_id

    # ── 读取 ──────────────────────────────────────────────────────────────────

    def get_active(self, user_id: str, min_importance: float = 0.1,
                   limit: int = 20) -> List[Dict[str, Any]]:
        """获取用户活跃的自适应记忆（按重要性降序）

        会先触发衰减计算，确保返回的分数是最新的。
        """
        self._apply_decay(user_id)
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, content, category, importance, reference_count,
                          created_at, last_referenced
                   FROM adaptive_memory
                   WHERE user_id = ? AND is_active = 1 AND importance >= ?
                   ORDER BY importance DESC
                   LIMIT ?""",
                (user_id, min_importance, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    # ── 引用（使用时加分）────────────────────────────────────────────────────

    def reference(self, memory_id: int) -> None:
        """标记一条记忆被引用，增加重要性"""
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE adaptive_memory
                   SET reference_count = reference_count + 1,
                       importance = MIN(1.0, importance + ?),
                       last_referenced = CURRENT_TIMESTAMP,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (REFERENCE_BOOST, memory_id)
            )

    # ── 时间衰减 ──────────────────────────────────────────────────────────────

    def _apply_decay(self, user_id: str) -> None:
        """对用户所有活跃记忆应用时间衰减

        衰减公式: importance = initial_importance * 2^(-days_elapsed / half_life)
                  + reference_bonus

        其中 reference_bonus = reference_count * REFERENCE_BOOST（上限0.3）
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, initial_importance, reference_count, created_at
                   FROM adaptive_memory
                   WHERE user_id = ? AND is_active = 1""",
                (user_id,)
            ).fetchall()

            now = datetime.now()
            updates = []
            deactivate_ids = []

            for row in rows:
                created = datetime.fromisoformat(row["created_at"])
                days_elapsed = (now - created).total_seconds() / 86400.0

                # 指数衰减
                decay_factor = math.pow(2, -days_elapsed / DECAY_HALF_LIFE_DAYS)
                base_importance = row["initial_importance"] * decay_factor

                # 引用加成（上限0.3）
                ref_bonus = min(0.3, row["reference_count"] * REFERENCE_BOOST)

                new_importance = min(1.0, base_importance + ref_bonus)

                if new_importance < MIN_IMPORTANCE:
                    deactivate_ids.append(row["id"])
                else:
                    updates.append((new_importance, row["id"]))

            # 批量更新
            if updates:
                conn.executemany(
                    "UPDATE adaptive_memory SET importance = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    updates
                )
            if deactivate_ids:
                placeholders = ",".join("?" * len(deactivate_ids))
                conn.execute(
                    f"UPDATE adaptive_memory SET is_active = 0 WHERE id IN ({placeholders})",
                    deactivate_ids
                )
                logger.info(
                    f"[AdaptiveMemory] 衰减清理: {len(deactivate_ids)} 条记忆因分数过低被停用"
                )

    # ── 定期总结/清理 ─────────────────────────────────────────────────────────

    def consolidate(self, user_id: str) -> Dict[str, int]:
        """定期总结与清理

        执行以下操作：
        1. 应用时间衰减
        2. 删除低分记忆
        3. 合并相似记忆（基于关键词重叠）
        4. 超出上限时保留高分记忆

        Returns:
            操作统计 {"decayed": n, "removed": n, "merged": n}
        """
        stats = {"decayed": 0, "removed": 0, "merged": 0}

        # 1. 应用衰减（内部会自动停用低分记忆）
        self._apply_decay(user_id)

        with self._get_conn() as conn:
            # 2. 统计被停用的数量
            removed = conn.execute(
                "SELECT COUNT(*) as cnt FROM adaptive_memory WHERE user_id = ? AND is_active = 0",
                (user_id,)
            ).fetchone()["cnt"]
            stats["removed"] = removed

            # 3. 合并相似记忆（简单策略：内容完全包含关系）
            active = conn.execute(
                """SELECT id, content, importance FROM adaptive_memory
                   WHERE user_id = ? AND is_active = 1
                   ORDER BY importance DESC""",
                (user_id,)
            ).fetchall()

            merged_ids = set()
            for i, mem_a in enumerate(active):
                if mem_a["id"] in merged_ids:
                    continue
                for j, mem_b in enumerate(active):
                    if i == j or mem_b["id"] in merged_ids:
                        continue
                    # 如果 B 的内容完全被 A 包含，则合并（停用 B）
                    if mem_b["content"] in mem_a["content"]:
                        merged_ids.add(mem_b["id"])

            if merged_ids:
                placeholders = ",".join("?" * len(merged_ids))
                conn.execute(
                    f"UPDATE adaptive_memory SET is_active = 0 WHERE id IN ({placeholders})",
                    list(merged_ids)
                )
                stats["merged"] = len(merged_ids)

            # 4. 强制上限
            self._enforce_limit(conn, user_id)

        logger.info(f"[AdaptiveMemory] 总结完成: {stats}")
        return stats

    def _enforce_limit(self, conn, user_id: str) -> None:
        """确保活跃记忆不超过上限，超出时停用最低分的"""
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM adaptive_memory WHERE user_id = ? AND is_active = 1",
            (user_id,)
        ).fetchone()["cnt"]

        if count > MAX_ADAPTIVE_MEMORIES:
            excess = count - MAX_ADAPTIVE_MEMORIES
            # 停用分数最低的 excess 条
            low_ids = conn.execute(
                """SELECT id FROM adaptive_memory
                   WHERE user_id = ? AND is_active = 1
                   ORDER BY importance ASC
                   LIMIT ?""",
                (user_id, excess)
            ).fetchall()
            if low_ids:
                ids = [r["id"] for r in low_ids]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE adaptive_memory SET is_active = 0 WHERE id IN ({placeholders})",
                    ids
                )
                logger.info(f"[AdaptiveMemory] 超出上限，停用 {len(ids)} 条低分记忆")

    # ── 格式化输出（用于注入 prompt）────────────────────────────────────────────

    def format_for_prompt(self, user_id: str, top_n: int = 10) -> str:
        """将用户的高分自适应记忆格式化为可注入 prompt 的文本

        Args:
            user_id: 用户标识
            top_n: 最多取前 N 条

        Returns:
            格式化的记忆文本，无记忆时返回空字符串
        """
        memories = self.get_active(user_id, min_importance=0.15, limit=top_n)
        if not memories:
            return ""

        lines = ["[系统观察到的用户习惯]"]
        for i, mem in enumerate(memories, 1):
            score_bar = "●" * int(mem["importance"] * 5) + "○" * (5 - int(mem["importance"] * 5))
            lines.append(f"{i}. {mem['content']} [{score_bar}]")

        return "\n".join(lines)

    # ── Agent 自动识别辅助 ────────────────────────────────────────────────────

    @staticmethod
    def assess_importance(content: str, context: Dict[str, Any] = None) -> float:
        """评估一条信息的初始重要性

        基于规则的快速评估（不调用 LLM）：
        - 包含具体数值/阈值的信息 → 高重要性
        - 包含偏好/习惯表达的信息 → 中高重要性
        - 一般性观察 → 中等重要性
        """
        score = 0.4  # 基础分

        # 包含具体数值（如"3年以上"、"50万"）
        import re
        if re.search(r'\d+[年万kK%]', content):
            score += 0.15

        # 包含偏好/频率词
        preference_words = ["经常", "总是", "喜欢", "偏好", "习惯", "倾向", "频繁", "每次"]
        if any(w in content for w in preference_words):
            score += 0.15

        # 包含否定偏好（"不喜欢"、"不要"）
        negative_words = ["不喜欢", "不要", "不看", "排除", "跳过", "忽略"]
        if any(w in content for w in negative_words):
            score += 0.1

        # 包含条件/规则表达
        rule_words = ["如果", "当", "只要", "前提", "条件", "必须"]
        if any(w in content for w in rule_words):
            score += 0.1

        return min(1.0, score)


# 全局单例
adaptive_memory = AdaptiveMemoryStore()
