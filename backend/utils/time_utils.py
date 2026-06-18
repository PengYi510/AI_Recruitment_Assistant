"""时间工具模块 - 获取当前北京时间

策略:
1. 优先通过NTP网络时间获取精确北京时间
2. 降级到HTTP时间API
3. 最终降级到系统本地时间（假定系统时区已正确设置为东八区）

提供缓存机制：时间获取结果缓存10分钟，避免频繁网络请求。
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# 北京时区 UTC+8
_BEIJING_TZ = timezone(timedelta(hours=8))

# 缓存：(timestamp, datetime_value)
_time_cache: Optional[tuple] = None
_CACHE_TTL = 600  # 缓存10分钟


def get_beijing_now() -> datetime:
    """获取当前北京时间（带缓存）

    优先级:
    1. NTP网络时间
    2. HTTP API时间
    3. 系统本地时间转北京时区

    Returns:
        北京时间的 datetime 对象（带时区信息）
    """
    global _time_cache

    # 检查缓存是否有效
    if _time_cache is not None:
        cached_ts, cached_dt = _time_cache
        elapsed = time.time() - cached_ts
        if elapsed < _CACHE_TTL:
            # 用缓存的基准时间 + 已过去的秒数推算当前时间
            return cached_dt + timedelta(seconds=elapsed)

    # 尝试各种方式获取时间
    beijing_time = _try_ntp_time()
    if beijing_time is None:
        beijing_time = _try_http_time()
    if beijing_time is None:
        beijing_time = _get_system_time()
        logger.info(f"[TimeUtils] Using system time: {beijing_time.isoformat()}")
    else:
        logger.info(f"[TimeUtils] Got network time: {beijing_time.isoformat()}")

    # 更新缓存
    _time_cache = (time.time(), beijing_time)
    return beijing_time


def get_current_year() -> int:
    """获取当前年份（北京时间）"""
    return get_beijing_now().year


def get_current_month() -> int:
    """获取当前月份（北京时间）"""
    return get_beijing_now().month


def _try_ntp_time() -> Optional[datetime]:
    """通过NTP获取网络时间"""
    try:
        import ntplib
        client = ntplib.NTPClient()
        # 尝试多个NTP服务器
        ntp_servers = [
            'ntp.aliyun.com',
            'cn.ntp.org.cn',
            'ntp.tencent.com',
            'pool.ntp.org',
        ]
        for server in ntp_servers:
            try:
                response = client.request(server, timeout=3)
                utc_time = datetime.fromtimestamp(response.tx_time, tz=timezone.utc)
                beijing_time = utc_time.astimezone(_BEIJING_TZ)
                logger.debug(f"[TimeUtils] NTP from {server}: {beijing_time}")
                return beijing_time
            except Exception:
                continue
    except ImportError:
        logger.debug("[TimeUtils] ntplib not installed, skipping NTP")
    return None


def _try_http_time() -> Optional[datetime]:
    """通过HTTP API获取网络时间"""
    try:
        import urllib.request
        import json

        # 尝试worldtimeapi
        apis = [
            ('http://worldtimeapi.org/api/timezone/Asia/Shanghai', 'datetime'),
            ('http://api.m.taobao.com/rest/api3.do?api=mtop.common.getTimestamp', None),
        ]

        for url, key in apis:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    if key == 'datetime':
                        # worldtimeapi 格式: "2026-06-13T11:30:00.123456+08:00"
                        dt_str = data.get('datetime', '')
                        if dt_str:
                            from datetime import datetime as dt_cls
                            beijing_time = dt_cls.fromisoformat(dt_str)
                            return beijing_time
                    elif key is None and 'data' in data:
                        # 淘宝时间戳API
                        ts = int(data['data'].get('t', 0)) / 1000
                        if ts > 0:
                            utc_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                            return utc_time.astimezone(_BEIJING_TZ)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _get_system_time() -> datetime:
    """获取系统本地时间并转换为北京时间"""
    # 获取当前UTC时间，转为北京时间
    utc_now = datetime.now(timezone.utc)
    return utc_now.astimezone(_BEIJING_TZ)
