"""
会话数据库存储模块

支持会话元信息和消息历史的持久化存储，
使用 SQLite 作为轻量级存储方案。
同时包含用户认证表（users）用于登录/注册功能。
"""

import sqlite3
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading

logger = logging.getLogger(__name__)


class SessionDatabase:
    """会话数据库管理器"""
    
    def __init__(self, db_path: str = "data/conversation_sessions.db"):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
    
    @staticmethod
    def _decode_metadata(metadata_raw: Any) -> Dict[str, Any]:
        """解析 metadata 字段，兼容空值和非法 JSON。"""
        if not metadata_raw:
            return {}
        if isinstance(metadata_raw, dict):
            return metadata_raw
        try:
            return json.loads(metadata_raw)
        except Exception:
            return {}

    @staticmethod
    def _merge_row_with_metadata(row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """将 metadata 中的扩展字段合并到行数据中。"""
        metadata = SessionDatabase._decode_metadata(row_dict.get("metadata"))
        merged = dict(row_dict)
        if isinstance(metadata, dict):
            for k, v in metadata.items():
                if k not in merged:
                    merged[k] = v
            merged["metadata"] = metadata
        else:
            merged["metadata"] = {}
        return merged

    def _init_db(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 创建用户表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    nickname TEXT DEFAULT '',
                    role TEXT DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )
            """)
            
            # 创建会话元信息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_mis TEXT NOT NULL,
                    title TEXT DEFAULT '新对话',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # 创建会话消息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
                )
            """)
            
            # 创建索引以提升查询性能
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_user 
                ON conversation_sessions(user_mis)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_session 
                ON conversation_messages(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_username
                ON users(username)
            """)
            
            conn.commit()

            # 初始化管理员账号（如果不存在）
            self._ensure_admin_account(conn)

            logger.info(f"数据库初始化完成: {self.db_path}")

    @staticmethod
    def _hash_password(password: str) -> str:
        """对密码进行 SHA256 哈希"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def _ensure_admin_account(self, conn):
        """确保管理员账号存在"""
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", ("admin",))
        if cursor.fetchone() is None:
            cursor.execute(
                """
                INSERT INTO users (username, password_hash, nickname, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("admin", self._hash_password("admin"), "管理员", "admin", datetime.now().isoformat()),
            )
            conn.commit()
            logger.info("已创建默认管理员账号: admin/admin")

    # ── 用户认证相关方法 ─────────────────────────────────────────────────────

    def register_user(self, username: str, password: str, nickname: str = "") -> Dict[str, Any]:
        """
        注册新用户

        Args:
            username: 用户名
            password: 密码（明文，内部会哈希）
            nickname: 昵称

        Returns:
            {"success": True, "user": {...}} 或 {"success": False, "error": "..."}
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    # 检查用户名是否已存在
                    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                    if cursor.fetchone() is not None:
                        return {"success": False, "error": "用户名已存在"}

                    now = datetime.now().isoformat()
                    cursor.execute(
                        """
                        INSERT INTO users (username, password_hash, nickname, role, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (username, self._hash_password(password), nickname or username, "user", now),
                    )
                    conn.commit()
                    user_id = cursor.lastrowid
                    return {
                        "success": True,
                        "user": {
                            "id": user_id,
                            "username": username,
                            "nickname": nickname or username,
                            "role": "user",
                            "created_at": now,
                        },
                    }
        except Exception as e:
            logger.error(f"注册用户失败: {e}")
            return {"success": False, "error": str(e)}

    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        验证用户登录

        Args:
            username: 用户名
            password: 密码（明文）

        Returns:
            {"success": True, "user": {...}} 或 {"success": False, "error": "..."}
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, username, password_hash, nickname, role, created_at FROM users WHERE username = ?",
                    (username,),
                )
                row = cursor.fetchone()
                if row is None:
                    return {"success": False, "error": "用户名或密码错误"}

                if row["password_hash"] != self._hash_password(password):
                    return {"success": False, "error": "用户名或密码错误"}

                # 更新最后登录时间
                now = datetime.now().isoformat()
                cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"]))
                conn.commit()

                return {
                    "success": True,
                    "user": {
                        "id": row["id"],
                        "username": row["username"],
                        "nickname": row["nickname"],
                        "role": row["role"],
                        "created_at": row["created_at"],
                    },
                }
        except Exception as e:
            logger.error(f"用户认证失败: {e}")
            return {"success": False, "error": str(e)}

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """根据用户名获取用户信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, username, nickname, role, created_at, last_login FROM users WHERE username = ?",
                    (username,),
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            return None
    
    def save_session(self, session_meta: Dict[str, Any]) -> bool:
        """
        保存或更新会话元信息

        Args:
            session_meta: 会话元信息字典，必须包含 session_id 和 user_mis

        Returns:
            是否保存成功
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    # 兼容扩展字段：除基础列之外的字段都塞入 metadata
                    base_fields = {
                        "session_id",
                        "user_mis",
                        "title",
                        "created_at",
                        "updated_at",
                        "message_count",
                        "metadata",
                    }
                    metadata_payload = self._decode_metadata(session_meta.get("metadata"))
                    for k, v in session_meta.items():
                        if k not in base_fields:
                            metadata_payload[k] = v

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO conversation_sessions
                        (session_id, user_mis, title, created_at, updated_at, message_count, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_meta.get("session_id"),
                            session_meta.get("user_mis"),
                            session_meta.get("title", "新对话"),
                            session_meta.get("created_at", datetime.now().isoformat()),
                            session_meta.get("updated_at", datetime.now().isoformat()),
                            session_meta.get("message_count", 0),
                            json.dumps(metadata_payload, ensure_ascii=False),
                        ),
                    )
                    conn.commit()
                    logger.info(f"保存会话: {session_meta.get('session_id')}")
                    return True
        except Exception as e:
            logger.error(f"保存会话失败: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话元信息

        Args:
            session_id: 会话ID

        Returns:
            会话元信息，不存在返回None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, user_mis, title, created_at, updated_at, message_count, metadata
                    FROM conversation_sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                if row:
                    return self._merge_row_with_metadata(dict(row))
                return None
        except Exception as e:
            logger.error(f"获取会话失败: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话及其所有消息
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 删除消息
                    cursor.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
                    
                    # 删除会话
                    cursor.execute("DELETE FROM conversation_sessions WHERE session_id = ?", (session_id,))
                    
                    conn.commit()
                    logger.info(f"删除会话: {session_id}")
                    return True
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            return False
    
    def list_user_sessions(self, user_mis: str) -> List[Dict[str, Any]]:
        """
        列出用户的所有会话（按更新时间倒序）

        Args:
            user_mis: 用户MIS

        Returns:
            会话列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, user_mis, title, created_at, updated_at, message_count, metadata
                    FROM conversation_sessions
                    WHERE user_mis = ?
                    ORDER BY updated_at DESC
                    """,
                    (user_mis,),
                )
                return [self._merge_row_with_metadata(dict(row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"列出用户会话失败: {e}")
            return []
    
    def save_messages(self, session_id: str, role: str, content: str) -> bool:
        """
        保存消息到数据库

        Args:
            session_id: 会话ID
            role: 消息角色 (user/assistant)
            content: 消息内容

        Returns:
            是否保存成功
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO conversation_messages
                        (id, session_id, role, content, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        None,  # 主键ID，None时由SQLite自增生成
                        session_id,
                        role,
                        content,
                        datetime.now().isoformat(),
                    ))
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"保存消息失败: {e}")
            return False
    
    def get_messages(self, session_id: str, limit: int = -1) -> List[Dict[str, Any]]:
        """
        获取会话的消息历史
        
        Args:
            session_id: 会话ID
            limit: 返回消息数量限制（-1表示全部）
            
        Returns:
            消息列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                    SELECT id, session_id, role, content, created_at
                    FROM conversation_messages
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                """
                
                if limit > 0:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, (session_id,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"获取会话消息失败: {e}")
            return []

    def get_all_messages(self, limit: int = -1) -> List[Dict[str, Any]]:
        """
        获取会话的消息历史
        
        Args:
            limit: 返回消息数量限制（-1表示全部）
            
        Returns:
            消息列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                    SELECT id, session_id, role, content, created_at
                    FROM conversation_messages
                    ORDER BY created_at ASC
                """
                
                if limit > 0:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, ())
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"获取会话消息失败: {e}")
            return []
    
    def update_last_assistant_message(self, session_id: str, new_content: str) -> bool:
        """
        将指定会话中最后一条 assistant 消息的内容更新为 new_content。
        用于流式输出场景：先写 placeholder，流式结束后回填真实内容。

        Args:
            session_id: 会话ID
            new_content: 替换后的消息内容

        Returns:
            是否更新成功
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        UPDATE conversation_messages
                        SET content = ?
                        WHERE id = (
                            SELECT id FROM conversation_messages
                            WHERE session_id = ? AND role = 'assistant'
                            ORDER BY id DESC
                            LIMIT 1
                        )
                        """,
                        (new_content, session_id),
                    )
                    conn.commit()
                    logger.info(f"回填最后一条 assistant 消息: session={session_id}, len={len(new_content)}")
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"回填 assistant 消息失败: {e}")
            return False

    def update_session_title(self, session_id: str, title: str) -> bool:
        """
        更新会话标题
        
        Args:
            session_id: 会话ID
            title: 新标题
            
        Returns:
            是否更新成功
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE conversation_sessions 
                        SET title = ?, updated_at = ?
                        WHERE session_id = ?
                    """, (title, datetime.now().isoformat(), session_id))
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"更新会话标题失败: {e}")
            return False
    
    def increment_message_count(self, session_id: str) -> bool:
        """
        递增会话消息计数
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否更新成功
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE conversation_sessions 
                        SET message_count = message_count + 1, updated_at = ?
                        WHERE session_id = ?
                    """, (datetime.now().isoformat(), session_id))
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"更新消息计数失败: {e}")
            return False
    
    def clear_old_sessions(self, days: int = 30) -> int:
        """
        清理N天前的会话数据（可选的定期维护任务）
        
        Args:
            days: 天数
            
        Returns:
            删除的会话数
        """
        try:
            from datetime import timedelta
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 获取要删除的会话ID
                    cursor.execute("""
                        SELECT session_id FROM conversation_sessions
                        WHERE updated_at < ?
                    """, (cutoff_date,))
                    session_ids = [row[0] for row in cursor.fetchall()]
                    
                    # 删除对应的消息
                    cursor.execute("""
                        DELETE FROM conversation_messages 
                        WHERE session_id IN (SELECT session_id FROM conversation_sessions WHERE updated_at < ?)
                    """, (cutoff_date,))
                    
                    # 删除会话
                    cursor.execute("""
                        DELETE FROM conversation_sessions WHERE updated_at < ?
                    """, (cutoff_date,))
                    
                    conn.commit()
                    logger.info(f"清理了 {len(session_ids)} 个旧会话")
                    return len(session_ids)
        except Exception as e:
            logger.error(f"清理旧会话失败: {e}")
            return 0

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """获取所有会话元信息（用于服务重启恢复）。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, user_mis, title, created_at, updated_at, message_count, metadata
                    FROM conversation_sessions
                    ORDER BY updated_at DESC
                    """
                )
                return [self._merge_row_with_metadata(dict(row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"获取全部会话失败: {e}")
            return []


# 全局数据库实例
session_db = SessionDatabase()

if __name__ == '__main__':
    print(session_db.get_all_sessions())
    print(session_db.get_all_messages())
    pass
