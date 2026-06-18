"""Layer 1: 显式长期记忆 (Persistent Memory)

存储用户明确要求记住的规则、偏好、定义。
特点：
- 仅当用户显式说"记住/remember/以后都..."时写入
- 仅当用户显式说"忘掉/取消/不再..."时删除或修改
- 不会自动衰减，永久有效
- 每条记忆有分类标签（rule/preference/definition）

数据表结构：
    persistent_memory (
        id          INTEGER PRIMARY KEY,
        user_id     TEXT NOT NULL,       -- 用户标识（MIS号）
        category    TEXT NOT NULL,       -- 分类: rule/preference/definition
        content     TEXT NOT NULL,       -- 记忆内容（自然语言描述）
        keywords    TEXT,                -- 关键词（逗号分隔，用于快速检索）
        created_at  TIMESTAMP,
        updated_at  TIMESTAMP,
        is_active   INTEGER DEFAULT 1    -- 软删除标记
    )
"""

from __future__ import annotations

import sqlite3
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from datetime import datetime

from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

# 记忆数据库路径（独立于业务数据库）
_MEMORY_DB_PATH = str(DATA_DIR / "memory.db")

# 注意：触发词检测和分类逻辑已迁移到 memory_loader.py
# 本模块仅负责 Layer 1 的 CRUD 存储操作


class PersistentMemoryStore:
    """显式长期记忆存储"""

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
        """初始化记忆表"""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS persistent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'rule',
                    content TEXT NOT NULL,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_pm_user_active
                    ON persistent_memory(user_id, is_active);
            """)

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def save(self, user_id: str, content: str, category: str = "rule",
             keywords: Optional[str] = None) -> int:
        """保存一条显式记忆

        Args:
            user_id: 用户标识
            content: 记忆内容（自然语言）
            category: 分类 (rule/preference/definition)
            keywords: 关键词（逗号分隔）

        Returns:
            新记录的 ID
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO persistent_memory (user_id, category, content, keywords)
                   VALUES (?, ?, ?, ?)""",
                (user_id, category, content, keywords)
            )
            record_id = cursor.lastrowid
            logger.info(f"[PersistentMemory] 保存记忆 #{record_id}: [{category}] {content[:50]}...")
            return record_id

    # ── 读取 ──────────────────────────────────────────────────────────────────

    def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户所有活跃的显式记忆"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, category, content, keywords, created_at, updated_at
                   FROM persistent_memory
                   WHERE user_id = ? AND is_active = 1
                   ORDER BY created_at ASC""",
                (user_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def search(self, user_id: str, keyword: str) -> List[Dict[str, Any]]:
        """按关键词搜索记忆"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, category, content, keywords, created_at
                   FROM persistent_memory
                   WHERE user_id = ? AND is_active = 1
                     AND (content LIKE ? OR keywords LIKE ?)
                   ORDER BY created_at DESC""",
                (user_id, f"%{keyword}%", f"%{keyword}%")
            ).fetchall()
            return [dict(row) for row in rows]

    # ── 修改 ──────────────────────────────────────────────────────────────────

    def update(self, memory_id: int, content: str,
               category: Optional[str] = None, keywords: Optional[str] = None) -> bool:
        """更新一条记忆的内容"""
        with self._get_conn() as conn:
            fields = ["content = ?", "updated_at = CURRENT_TIMESTAMP"]
            params: list = [content]
            if category:
                fields.append("category = ?")
                params.append(category)
            if keywords is not None:
                fields.append("keywords = ?")
                params.append(keywords)
            params.append(memory_id)
            conn.execute(
                f"UPDATE persistent_memory SET {', '.join(fields)} WHERE id = ?",
                params
            )
            logger.info(f"[PersistentMemory] 更新记忆 #{memory_id}")
            return True

    def deactivate(self, memory_id: int) -> bool:
        """软删除一条记忆"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE persistent_memory SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (memory_id,)
            )
            logger.info(f"[PersistentMemory] 停用记忆 #{memory_id}")
            return True

    def deactivate_by_keyword(self, user_id: str, keyword: str) -> int:
        """按关键词软删除匹配的记忆，返回影响行数"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """UPDATE persistent_memory
                   SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = ? AND is_active = 1
                     AND (content LIKE ? OR keywords LIKE ?)""",
                (user_id, f"%{keyword}%", f"%{keyword}%")
            )
            count = cursor.rowcount
            if count > 0:
                logger.info(f"[PersistentMemory] 按关键词'{keyword}'停用 {count} 条记忆")
            return count

    # ── 格式化输出（用于注入 prompt）────────────────────────────────────────────

    def format_for_prompt(self, user_id: str) -> str:
        """将用户的所有活跃记忆格式化为可注入 prompt 的文本

        Returns:
            格式化的记忆文本，无记忆时返回空字符串
        """
        memories = self.get_all(user_id)
        if not memories:
            return ""

        lines = ["[用户长期偏好与规则]"]
        for i, mem in enumerate(memories, 1):
            cat_label = {"rule": "规则", "preference": "偏好", "definition": "定义"}.get(
                mem["category"], "其他"
            )
            lines.append(f"{i}. [{cat_label}] {mem['content']}")

        return "\n".join(lines)



# 全局单例
persistent_memory = PersistentMemoryStore()
