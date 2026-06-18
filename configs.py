"""应用配置模块

提供白名单等基础配置，可通过环境变量覆盖。
"""

import os


# ── 白名单配置 ────────────────────────────────────────────────────────────────

# 有下载权限的用户 MIS 列表
DOWNLOAD_WHITELIST = set(
    os.environ.get("DOWNLOAD_WHITELIST", "admin,pengyi14").split(",")
)

# 有标签查看权限的用户 MIS 列表
TAG_WHITELIST = set(
    os.environ.get("TAG_WHITELIST", "admin,pengyi14").split(",")
)
