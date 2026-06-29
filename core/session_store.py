"""SessionStore - Redis 实现（生产级），Redis 不可用时自动降级为 SQLite 持久化。

使用方式：
    from core.session_store import session_store

    session_store.set("key", {"data": 1}, ttl=3600)  # ttl 单位秒，None 表示永不过期
    value = session_store.get("key")                  # 返回原始对象，过期或不存在返回 None
    session_store.delete("key")
    session_store.exists("key")                       # True / False

Redis 配置（环境变量）：
    REDIS_HOST      默认 127.0.0.1
    REDIS_PORT      默认 6379
    REDIS_DB        默认 0
    REDIS_PASSWORD  默认 None（无密码）

降级策略：
    Redis 不可用时自动降级为 SQLite 持久化实现（SQLiteSessionStore），
    数据存储在 data/sqlite/session_store.db 中，重启后可恢复。
    日志中会打印 WARNING 提示。上层代码无需关心底层使用的是哪种实现。

Session 数据结构约定（由 http_server.py 维护）：
    {
        "history": [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "pending_interaction": None | {
            "interaction_id":  str,
            "source":          str,
            "stage":           int,
            "interaction_type": "confirm" | "select" | "input",
            "prompt":          str,
            "options":         list,
            "default":         any,
            # 恢复执行所需的上下文：
            "resume_context": {
                "resolved_query":  str,
                "selected_agents": list[str],
                "entities":        dict,
                "permission":      str,
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Redis 实现
# ═══════════════════════════════════════════════════════════════════════════════

class RedisSessionStore:
    """
    基于 Redis 的 SessionStore。

    所有值以 JSON 序列化后存储，支持 TTL 自动过期。
    接口与 SQLiteSessionStore 完全一致，上层零改动。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ) -> None:
        import redis
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,      # 返回 str 而非 bytes
            socket_connect_timeout=3,   # 连接超时3秒
            socket_timeout=5,           # 操作超时5秒
        )
        # 验证连接可用
        self._client.ping()
        logger.info(f"[SessionStore] Redis 连接成功: {host}:{port} db={db}")

    # ── 核心接口 ────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """读取 key 的值，不存在或已过期返回 None。"""
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        写入 key，可选 TTL（秒）。
        ttl=None  → 永不过期（Redis PERSIST 语义）
        ttl=3600  → 1小时后过期（Redis SET key value EX 3600）
        """
        serialized = json.dumps(value, ensure_ascii=False)
        if ttl is not None:
            self._client.setex(key, ttl, serialized)
        else:
            self._client.set(key, serialized)

    def delete(self, key: str) -> bool:
        """删除 key，返回是否存在。"""
        return self._client.delete(key) > 0

    def exists(self, key: str) -> bool:
        """key 是否存在且未过期。"""
        return self._client.exists(key) > 0

    def expire(self, key: str, ttl: int) -> bool:
        """为已存在的 key 设置/重置 TTL（秒）。"""
        return self._client.expire(key, ttl)

    def ttl(self, key: str) -> int:
        """
        返回剩余 TTL 秒数。
        -2: key 不存在
        -1: key 存在但无过期时间
        >=0: 剩余秒数
        """
        return self._client.ttl(key)


# ═══════════════════════════════════════════════════════════════════════════════
#  SQLite Fallback 实现（Redis 不可用时自动降级）
# ═══════════════════════════════════════════════════════════════════════════════

class SQLiteSessionStore:
    """
    基于 SQLite 的 SessionStore（降级方案），线程安全，支持 TTL，数据持久化。

    数据存储在 data/sqlite/session_store.db 中，服务重启后 session 不丢失。
    过期条目在读取时懒清理，同时有后台线程定期清理过期数据。
    仅在 Redis 不可用时使用。
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS session_kv (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        expire_at  REAL             -- NULL 表示永不过期，否则为 Unix 时间戳
    );
    CREATE INDEX IF NOT EXISTS idx_session_kv_expire
        ON session_kv(expire_at)
        WHERE expire_at IS NOT NULL;
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            # 默认放在项目 data/sqlite/ 目录下
            project_root = Path(__file__).parent.parent
            sqlite_dir = project_root / "data" / "sqlite"
            sqlite_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(sqlite_dir / "session_store.db")

        self._db_path = db_path
        self._lock = threading.Lock()

        # 初始化数据库
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

        # 启动后台清理线程（每 5 分钟清理一次过期条目）
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True,
            name="session-store-cleanup",
        )
        self._cleanup_thread.start()

        logger.info(f"[SessionStore] SQLite 持久化存储初始化成功: {db_path}")

    def _connect(self) -> sqlite3.Connection:
        """创建线程局部的 SQLite 连接。"""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # ── 核心接口 ────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """读取 key 的值，不存在或已过期返回 None。"""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value, expire_at FROM session_kv WHERE key = ?",
                    (key,),
                ).fetchone()

                if row is None:
                    return None

                value_str, expire_at = row
                if expire_at is not None and time.time() > expire_at:
                    # 惰性清理过期条目
                    conn.execute("DELETE FROM session_kv WHERE key = ?", (key,))
                    return None

                try:
                    return json.loads(value_str)
                except (json.JSONDecodeError, TypeError):
                    return value_str

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        写入 key，可选 TTL（秒）。
        ttl=None  → 永不过期
        ttl=7200  → 2小时后过期
        """
        serialized = json.dumps(value, ensure_ascii=False)
        expire_at = (time.time() + ttl) if ttl is not None else None
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO session_kv (key, value, expire_at) "
                    "VALUES (?, ?, ?)",
                    (key, serialized, expire_at),
                )

    def delete(self, key: str) -> bool:
        """删除 key，返回是否存在。"""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM session_kv WHERE key = ?", (key,)
                )
                return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        """key 是否存在且未过期。"""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT expire_at FROM session_kv WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return False
                expire_at = row[0]
                if expire_at is not None and time.time() > expire_at:
                    conn.execute("DELETE FROM session_kv WHERE key = ?", (key,))
                    return False
                return True

    def expire(self, key: str, ttl: int) -> bool:
        """为已存在的 key 设置/重置 TTL（秒）。"""
        with self._lock:
            with self._connect() as conn:
                # 先检查 key 是否存在且未过期
                row = conn.execute(
                    "SELECT expire_at FROM session_kv WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return False
                expire_at = row[0]
                if expire_at is not None and time.time() > expire_at:
                    conn.execute("DELETE FROM session_kv WHERE key = ?", (key,))
                    return False
                # 设置新的过期时间
                new_expire = time.time() + ttl
                conn.execute(
                    "UPDATE session_kv SET expire_at = ? WHERE key = ?",
                    (new_expire, key),
                )
                return True

    def ttl(self, key: str) -> int:
        """
        返回剩余 TTL 秒数。
        -2: key 不存在
        -1: key 存在但无过期时间
        >=0: 剩余秒数
        """
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT expire_at FROM session_kv WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return -2
                expire_at = row[0]
                if expire_at is None:
                    return -1
                remaining = expire_at - time.time()
                if remaining <= 0:
                    conn.execute("DELETE FROM session_kv WHERE key = ?", (key,))
                    return -2
                return max(0, int(remaining))

    # ── 后台清理 ────────────────────────────────────────────────────────────

    def _periodic_cleanup(self) -> None:
        """后台线程：每 5 分钟清理一次过期条目。"""
        while True:
            try:
                time.sleep(300)  # 5 分钟
                self._cleanup_expired()
            except Exception:  # noqa: BLE001
                pass  # 后台线程不应因异常退出

    def _cleanup_expired(self) -> None:
        """批量删除所有已过期的条目。"""
        now = time.time()
        with self._lock:
            try:
                with self._connect() as conn:
                    cursor = conn.execute(
                        "DELETE FROM session_kv WHERE expire_at IS NOT NULL AND expire_at < ?",
                        (now,),
                    )
                    if cursor.rowcount > 0:
                        logger.debug(
                            f"[SessionStore] SQLite 清理了 {cursor.rowcount} 条过期 session"
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[SessionStore] SQLite 清理过期条目失败: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  全局单例（自动选择 Redis 或 SQLite 降级）
# ═══════════════════════════════════════════════════════════════════════════════

def _create_session_store():
    """尝试连接 Redis，失败则降级为 SQLite 持久化实现。"""
    redis_host = os.environ.get("REDIS_HOST", "127.0.0.1")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    redis_db = int(os.environ.get("REDIS_DB", "0"))
    redis_password = os.environ.get("REDIS_PASSWORD", None)

    try:
        store = RedisSessionStore(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
        )
        return store
    except Exception as e:
        logger.warning(
            f"[SessionStore] Redis 连接失败 ({redis_host}:{redis_port}): {e}. "
            f"降级为 SQLite 持久化实现（SQLiteSessionStore），数据存储在本地数据库。"
        )
        return SQLiteSessionStore()


session_store = _create_session_store()


# ── Session Key 规范 ─────────────────────────────────────────────────────────

SESSION_TTL = 3600 * 2  # 对话 session 默认保留 2 小时


def session_key(session_id: str) -> str:
    """对话历史 + 挂起交互的 session key"""
    return f"aibp:session:{session_id}"


def talent_page_key(session_id: str) -> str:
    """talent_recommend 分页上下文的 session key。

    存储结构：
    {
        "page_no":    int,   # 当前已返回的页码
        "session_id": str,   # 原始推荐请求的 sessionId（翻页时透传给服务端）
        "message_id": str,   # 原始推荐请求的 messageId
        "req_type":   int,   # 请求类型（通常为 1）
        "empl_id":    str,   # 发起请求的员工工号
        "user_mis":   str,   # 发起请求的用户 MIS
    }
    """
    return f"aibp:talent_page:{session_id}"
