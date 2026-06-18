"""短期记忆模块 - 读取用户最近的对话记忆摘要

从 data/memory/{user_mis}/ 目录下读取今天和昨天的 .md 文件，
合并后返回近期记忆上下文，用于增强后续对话的上下文理解能力。

若目录/文件不存在，返回空字符串（不报错）。
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 记忆文件存储根目录
MEMORY_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "memory"

# 内存缓存
_cache: dict = {}


def get_short_memory_context(
    force_reload: bool = False,
    user_mis: Optional[str] = None,
) -> str:
    """获取用户近期短期记忆内容（今天 + 昨天）

    Args:
        force_reload: 是否强制重新读取文件（忽略缓存）
        user_mis: 用户 MIS 标识，用于隔离不同用户的记忆

    Returns:
        合并后的记忆内容字符串，无记忆时返回空字符串
    """
    if not user_mis:
        return ""

    cache_key = f"{user_mis}"
    if not force_reload and cache_key in _cache:
        return _cache[cache_key]

    user_memory_dir = MEMORY_DIR / user_mis
    if not user_memory_dir.exists():
        _cache[cache_key] = ""
        return ""

    # 读取今天和昨天的记忆文件
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    date_strs = [
        yesterday.strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
    ]

    contents = []
    for date_str in date_strs:
        md_file = user_memory_dir / f"{date_str}.md"
        if md_file.exists():
            try:
                text = md_file.read_text(encoding="utf-8").strip()
                if text:
                    contents.append(f"## {date_str}\n{text}")
            except Exception as e:
                logger.warning(f"读取记忆文件失败 {md_file}: {e}")

    result = "\n\n".join(contents)
    _cache[cache_key] = result
    return result


def save_short_memory(user_mis: str, content: str) -> None:
    """保存当天的短期记忆

    Args:
        user_mis: 用户 MIS 标识
        content: 要追加的记忆内容
    """
    if not user_mis or not content.strip():
        return

    user_memory_dir = MEMORY_DIR / user_mis
    user_memory_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y-%m-%d")
    md_file = user_memory_dir / f"{today_str}.md"

    try:
        existing = ""
        if md_file.exists():
            existing = md_file.read_text(encoding="utf-8")
        with open(md_file, "w", encoding="utf-8") as f:
            if existing:
                f.write(existing.rstrip() + "\n\n")
            f.write(content.strip() + "\n")
        # 清除缓存
        _cache.pop(user_mis, None)
        logger.info(f"短期记忆已保存: {md_file}")
    except Exception as e:
        logger.error(f"保存短期记忆失败: {e}")
