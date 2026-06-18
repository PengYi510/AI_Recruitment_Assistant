"""统一日志配置模块

使用方式：
    在服务入口（http_server.py / frontend_server.py）的最顶部调用：

        from core.logging_config import setup_logging
        setup_logging()   # 会自动读取 SERVICE_NAME 环境变量

日志文件路径规则：
    SERVICE_NAME=http_server     → logs/http_server/app.log
    SERVICE_NAME=frontend_server → logs/frontend_server/app.log
    未设置                        → logs/app.log（仅控制台）

日志格式：
    %(asctime)s [%(levelname)s] [%(name)s] %(message)s
    示例：2026-05-08 10:00:00,123 [INFO] [agents.main_agent] plan: {...}

第三方库噪声抑制：
    pycat / squirrel / asyncio / urllib3 / apscheduler / pylion → WARNING 级别

日志去重机制：
    configs.py 中 pylion.init_logging() 内部调用 logging.config.dictConfig()，
    dictConfig 会直接替换 root_logger.handlers 列表（绕过 addHandler），
    导致第三方 FileHandler 被注入，日志重复输出。

    解决方案：在 root logger 的 callHandlers 层面做过滤，
    只让 AIBP 自有的 handler 处理日志记录，无论第三方用什么方式注入 handler。
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── 噪声库：统一调高到 WARNING，避免刷屏 ─────────────────────────────────────
_NOISY_LOGGERS = [
    "pycat",
    "squirrel.squirrelClient",
    "asyncio",
    "urllib3",
    "apscheduler",
    "pylion",
    "werkzeug",          # Flask 默认请求日志，可按需调整
    "octo-rpc",          # octo RPC 框架 TRACING 调试日志
    "octo_rpc",          # 同上（部分版本用下划线命名）
]

_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _AIBPHandler(logging.Handler):
    """标记 handler 为 AIBP 自有，用于区分第三方库偷偷注册的 handler。

    此 handler 本身不处理任何日志，仅作为基类/标记使用。
    所有通过 setup_logging 注册的 handler 都会继承此类，
    以便 callHandlers 过滤器识别并放行。
    """
    pass


def _make_aibp_handler(cls_name: str, base_cls: type) -> type:
    """动态创建一个同时继承 _AIBPHandler 和 base_cls 的新类。"""
    return type(cls_name, (_AIBPHandler, base_cls), {})


def setup_logging(
    level: int = logging.INFO,
    log_dir: str | None = None,
) -> None:
    """
    初始化全局日志配置，幂等调用（重复调用不会重复添加 handler）。
    通过 monkey-patch root logger 的 callHandlers 方法，在日志分发层面
    过滤掉非 AIBP handler，彻底防止第三方库（如 pylion init_logging 内部
    的 dictConfig）注入 handler 导致日志重复。

    Parameters
    ----------
    level:
        根 logger 的日志级别，默认 INFO（DEBUG 日志不记录）。
    log_dir:
        日志目录覆盖。若为 None，则从环境变量 SERVICE_NAME 自动推断：
          SERVICE_NAME=http_server     → ./logs/http_server/app.log
          SERVICE_NAME=frontend_server → logs/frontend_server/app.log
          未设置                        → 仅控制台，不写文件
    """
    root_logger = logging.getLogger()

    # 幂等：用自定义属性标记
    if getattr(root_logger, "_aibp_configured", False):
        return
    root_logger._aibp_configured = True  # type: ignore[attr-defined]

    # 清除已有的第三方 handler
    root_logger.handlers.clear()

    root_logger.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── 文件 Handler（按 10 MB 滚动，保留 5 份） ──────────────────────────────
    # 注意：start.sh 通过 ">> app.log 2>&1" 将 stdout/stderr 重定向到 app.log，
    # 若同时注册 console_handler（写 stdout）和 file_handler（直接写 app.log），
    # 则每条日志会被 shell 重定向 + file_handler 各写一次，导致双份输出。
    # 因此：有 file_handler 时不再注册 console_handler，避免双重写入。
    service_name = os.environ.get("SERVICE_NAME", "")
    if log_dir is None and service_name:
        log_dir = os.path.join("logs", service_name)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(log_dir, "app.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.__class__ = _make_aibp_handler("_AIBPFileHandler", RotatingFileHandler)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    else:
        # ── 无文件 Handler 时，才添加控制台 Handler（本地开发用）────────────────
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.__class__ = _make_aibp_handler("_AIBPConsoleHandler", logging.StreamHandler)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        console_handler.setStream(sys.stdout)
        root_logger.addHandler(console_handler)

    # ════════════════════════════════════════════════════════════════════════
    #  核心：monkey-patch root logger 的 callHandlers 方法
    #
    #  logging.Logger.handle() 调用 self.callHandlers(record) 来分发日志，
    #  callHandlers 遍历 self.handlers 列表，逐个调用 handler.handle(record)。
    #
    #  pylion.init_logging() 内部的 dictConfig 直接替换 self.handlers 列表，
    #  绕过 addHandler。但无论 handlers 列表被怎么篡改，
    #  最终都要经过 callHandlers 来分发。
    #
    #  我们在这里做最后一道过滤：只让 AIBP handler 处理日志。
    # ════════════════════════════════════════════════════════════════════════
    _original_callHandlers = root_logger.callHandlers

    def _filtered_callHandlers(record):
        """只通过 AIBP 自有 handler 分发日志，忽略第三方注入的 handler。"""
        # 保存原始 handlers 列表，临时替换为过滤后的列表
        original_handlers = root_logger.handlers[:]
        # 过滤：只保留 AIBP handler
        root_logger.handlers = [h for h in original_handlers if isinstance(h, _AIBPHandler)]
        try:
            _original_callHandlers(record)
        finally:
            # 恢复原始列表（不影响外部看到的 handlers 状态）
            root_logger.handlers = original_handlers

    root_logger.callHandlers = _filtered_callHandlers  # type: ignore[assignment]

    # 同时也守卫 addHandler 作为双重保险
    _original_addHandler = root_logger.addHandler

    def _guarded_addHandler(handler: logging.Handler) -> None:
        if isinstance(handler, _AIBPHandler):
            _original_addHandler(handler)
        # else: 非AIBP handler，静默丢弃

    root_logger.addHandler = _guarded_addHandler  # type: ignore[assignment]

    # ── 压制噪声第三方库 ──────────────────────────────────────────────────────
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
