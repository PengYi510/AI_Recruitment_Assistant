"""SessionStore - Redis 实现（生产级），Redis 不可用时自动降级为内存版。

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

如果 Redis 连接失败，系统自动降级为线程安全的内存实现（InMemorySessionStore），
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
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Redis 实现
# ═══════════════════════════════════════════════════════════════════════════════

class RedisSessionStore:
    """
    基于 Redis 的 SessionStore。

    所有值以 JSON 序列化后存储，支持 TTL 自动过期。
    接口与旧版 InMemorySessionStore 完全一致，上层零改动。
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
#  内存 Fallback 实现（Redis 不可用时自动降级）
# ═══════════════════════════════════════════════════════════════════════════════

class InMemorySessionStore:
    """
    内存版 SessionStore（降级方案），线程安全，支持 TTL。
    仅在 Redis 不可用时使用。
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._expire_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if self._is_expired(key):
                self._evict(key)
                return None
            return self._data.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            self._data[key] = value
            if ttl is not None:
                self._expire_at[key] = time.time() + ttl
            else:
                self._expire_at.pop(key, None)

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = key in self._data
            self._evict(key)
            return existed

    def exists(self, key: str) -> bool:
        with self._lock:
            if self._is_expired(key):
                self._evict(key)
                return False
            return key in self._data

    def expire(self, key: str, ttl: int) -> bool:
        with self._lock:
            if key not in self._data or self._is_expired(key):
                return False
            self._expire_at[key] = time.time() + ttl
            return True

    def ttl(self, key: str) -> int:
        with self._lock:
            if key not in self._data:
                return -2
            if self._is_expired(key):
                self._evict(key)
                return -2
            if key not in self._expire_at:
                return -1
            return max(0, int(self._expire_at[key] - time.time()))

    def _is_expired(self, key: str) -> bool:
        expire_time = self._expire_at.get(key)
        return expire_time is not None and time.time() > expire_time

    def _evict(self, key: str) -> None:
        self._data.pop(key, None)
        self._expire_at.pop(key, None)


# ═══════════════════════════════════════════════════════════════════════════════
#  全局单例（自动选择 Redis 或内存降级）
# ═══════════════════════════════════════════════════════════════════════════════

def _create_session_store():
    """尝试连接 Redis，失败则降级为内存实现。"""
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
            f"降级为内存实现（InMemorySessionStore），数据不持久化。"
        )
        return InMemorySessionStore()


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
