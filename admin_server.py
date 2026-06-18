"""开发者管理面板服务器

功能:
- 查看已入库简历（超级管理员可 CRUD）
- 查看待入库简历（支持多选手动入库）
- LLM 配置编辑
- 系统日志查看
- 用户管理（超级管理员 + 普通管理员）

端口: 9035 起，如被占用自动递增（9036, 9037...）
超级管理员: admin / admin
"""

import os
import sys
import json
import time
import socket
import hashlib
import logging
import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    PROJECT_ROOT as CONFIG_PROJECT_ROOT, SQLITE_DB_PATH
)
from backend.resume_scanner import (
    get_pending_resumes, get_scanner_status,
    manual_scan, manual_ingest_files,
    ScannerState, RESUME_DATA_DIR
)

logger = logging.getLogger(__name__)

# ====================================================================
# 管理面板配置
# ====================================================================

ADMIN_PORT_START = 9035
ADMIN_DB_PATH = str(CONFIG_PROJECT_ROOT / "data" / "admin_users.db")
LOG_DIR = CONFIG_PROJECT_ROOT / "logs"

# ====================================================================
# 用户数据库
# ====================================================================

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _init_admin_db():
    Path(ADMIN_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ADMIN_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'admin',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    cursor = conn.execute("SELECT id FROM admin_users WHERE username = 'admin'")
    if not cursor.fetchone():
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", _hash_password("admin"), "super_admin")
        )
    conn.commit()
    conn.close()


def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(ADMIN_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM admin_users WHERE username = ? AND password_hash = ?",
        (username, _hash_password(password))
    )
    row = cursor.fetchone()
    if row:
        conn.execute(
            "UPDATE admin_users SET last_login = ? WHERE id = ?",
            (datetime.now().isoformat(), row["id"])
        )
        conn.commit()
        conn.close()
        return dict(row)
    conn.close()
    return None


# 角色体系：
#   super_admin 超级管理员（系统内置，唯一，可管理一切）
#   admin       普通管理员（可登录后台、管理简历）
#   user        普通用户（只读/受限账号，非管理员）
VALID_ROLES = ("super_admin", "admin", "user")
ADMIN_ROLES = ("super_admin", "admin")


def register_user(username: str, password: str, role: str = "user") -> Dict[str, Any]:
    """创建账号。默认创建普通用户（user）。

    role 可选 'admin'（普通管理员）或 'user'（普通用户）；
    不允许通过此接口创建 super_admin。
    """
    if role not in ("admin", "user"):
        role = "user"
    conn = sqlite3.connect(ADMIN_DB_PATH)
    try:
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, _hash_password(password), role)
        )
        conn.commit()
        return {"success": True, "message": "创建成功", "role": role}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "用户名已存在"}
    finally:
        conn.close()


def get_all_users() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(ADMIN_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT id, username, role, created_at, last_login FROM admin_users "
        "ORDER BY CASE role WHEN 'super_admin' THEN 0 WHEN 'admin' THEN 1 ELSE 2 END, id"
    )
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users


def get_users_grouped() -> Dict[str, List[Dict[str, Any]]]:
    """返回按角色分组的账号名单：管理员名单 + 普通用户名单。"""
    all_users = get_all_users()
    admins = [u for u in all_users if u["role"] in ADMIN_ROLES]
    normal_users = [u for u in all_users if u["role"] == "user"]
    return {
        "admins": admins,
        "users": normal_users,
        "admin_count": len(admins),
        "user_count": len(normal_users),
    }


def delete_user(user_id: int) -> bool:
    conn = sqlite3.connect(ADMIN_DB_PATH)
    cursor = conn.execute("SELECT role FROM admin_users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    if not row or row[0] == "super_admin":
        conn.close()
        return False
    conn.execute("DELETE FROM admin_users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True


# ====================================================================
# FastAPI 应用
# ====================================================================

app = FastAPI(title="HR Agent 开发者管理面板", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, Dict[str, Any]] = {}


# ── Pydantic 模型 ──

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "user"  # 'admin' 或 'user'，默认普通用户


class IngestRequest(BaseModel):
    file_paths: List[str]


class LLMConfigUpdate(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


# ── 认证 ──

def get_current_user(request: Request) -> Dict[str, Any]:
    token = request.headers.get("X-Admin-Token", "")
    if token and token in _sessions:
        return _sessions[token]
    raise HTTPException(status_code=401, detail="未登录或会话已过期")


def require_super_admin(user: Dict[str, Any] = Depends(get_current_user)):
    if user.get("role") != "super_admin":
        raise HTTPException(status_code=403, detail="需要超级管理员权限")
    return user


# ── 认证接口 ──

@app.post("/api/login")
def login(req: LoginRequest):
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = hashlib.sha256(f"{req.username}{time.time()}".encode()).hexdigest()
    _sessions[token] = {"username": user["username"], "role": user["role"], "id": user["id"]}
    return {"token": token, "username": user["username"], "role": user["role"]}


@app.post("/api/register")
def register(req: RegisterRequest):
    """公开注册入口：仅允许创建普通用户（user），不能自助注册管理员。"""
    if len(req.username) < 3 or len(req.password) < 3:
        raise HTTPException(status_code=400, detail="用户名和密码至少3个字符")
    result = register_user(req.username, req.password, role="user")
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"  # 超管创建时可指定 'admin' 或 'user'


@app.post("/api/users")
def create_user(req: CreateUserRequest, user: Dict = Depends(require_super_admin)):
    """超级管理员创建账号，可指定角色为普通管理员(admin)或普通用户(user)。"""
    if len(req.username) < 3 or len(req.password) < 3:
        raise HTTPException(status_code=400, detail="用户名和密码至少3个字符")
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="角色只能是 admin 或 user")
    result = register_user(req.username, req.password, role=req.role)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/logout")
def logout(request: Request):
    token = request.headers.get("X-Admin-Token", "")
    _sessions.pop(token, None)
    return {"success": True}


# ── 简历管理接口 ──

@app.get("/api/resumes/ingested")
def get_ingested_resumes(
    page: int = 1,
    page_size: int = 20,
    search: str = "",
    user: Dict = Depends(get_current_user)
):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    offset = (page - 1) * page_size

    if search:
        like_pattern = f"%{search}%"
        query = """
            SELECT c.id, c.name, c.gender, c.phone, c.email,
                   COALESCE((
                       SELECT w.position FROM work_experiences w
                       WHERE w.candidate_id = c.id AND w.position IS NOT NULL AND w.position != ''
                       ORDER BY w.id DESC LIMIT 1
                   ), c.major) AS current_position,
                   c.work_years,
                   c.education_level AS highest_education,
                   c.current_city AS location,
                   c.created_at
            FROM candidates c
            WHERE c.name LIKE ? OR c.email LIKE ? OR c.major LIKE ?
            ORDER BY c.created_at DESC LIMIT ? OFFSET ?
        """
        cursor = conn.execute(query, (like_pattern, like_pattern, like_pattern, page_size, offset))
        total = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE name LIKE ? OR email LIKE ? OR major LIKE ?",
            (like_pattern, like_pattern, like_pattern)
        ).fetchone()[0]
    else:
        query = """
            SELECT c.id, c.name, c.gender, c.phone, c.email,
                   COALESCE((
                       SELECT w.position FROM work_experiences w
                       WHERE w.candidate_id = c.id AND w.position IS NOT NULL AND w.position != ''
                       ORDER BY w.id DESC LIMIT 1
                   ), c.major) AS current_position,
                   c.work_years,
                   c.education_level AS highest_education,
                   c.current_city AS location,
                   c.created_at
            FROM candidates c ORDER BY c.created_at DESC LIMIT ? OFFSET ?
        """
        cursor = conn.execute(query, (page_size, offset))
        total = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    resumes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"resumes": resumes, "total": total, "page": page, "page_size": page_size}


@app.get("/api/resumes/detail/{candidate_id}")
def get_resume_detail(candidate_id: int, user: Dict = Depends(get_current_user)):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    candidate = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if not candidate:
        conn.close()
        raise HTTPException(status_code=404, detail="简历不存在")
    result = dict(candidate)
    edu = conn.execute("SELECT * FROM education_history WHERE candidate_id = ?", (candidate_id,)).fetchall()
    result["education"] = [dict(e) for e in edu]
    work = conn.execute("SELECT * FROM work_experiences WHERE candidate_id = ?", (candidate_id,)).fetchall()
    result["work_experiences"] = [dict(w) for w in work]
    skills = conn.execute("SELECT * FROM candidate_skills WHERE candidate_id = ?", (candidate_id,)).fetchall()
    result["skills"] = [dict(s) for s in skills]
    projects = conn.execute("SELECT * FROM projects WHERE candidate_id = ?", (candidate_id,)).fetchall()
    result["projects"] = [dict(p) for p in projects]
    conn.close()
    return result


@app.delete("/api/resumes/{candidate_id}")
def delete_resume(candidate_id: int, user: Dict = Depends(require_super_admin)):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.execute("SELECT id FROM candidates WHERE id = ?", (candidate_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="简历不存在")
    conn.execute("DELETE FROM education_history WHERE candidate_id = ?", (candidate_id,))
    conn.execute("DELETE FROM work_experiences WHERE candidate_id = ?", (candidate_id,))
    conn.execute("DELETE FROM candidate_skills WHERE candidate_id = ?", (candidate_id,))
    conn.execute("DELETE FROM projects WHERE candidate_id = ?", (candidate_id,))
    conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": f"已删除候选人 {candidate_id}"}


# ── 待入库简历接口 ──

@app.get("/api/resumes/pending")
def get_pending(user: Dict = Depends(get_current_user)):
    return {"files": get_pending_resumes()}


@app.post("/api/resumes/ingest")
async def ingest_resumes(req: IngestRequest, user: Dict = Depends(require_super_admin)):
    if not req.file_paths:
        raise HTTPException(status_code=400, detail="请选择至少一个文件")
    results = await manual_ingest_files(req.file_paths)
    return {"results": results}


@app.post("/api/resumes/scan")
async def trigger_scan(user: Dict = Depends(require_super_admin)):
    result = await manual_scan()
    return result


# ── 扫描器状态接口 ──

@app.get("/api/scanner/status")
def scanner_status(user: Dict = Depends(get_current_user)):
    return get_scanner_status()


# ── LLM 配置接口 ──

@app.get("/api/config/llm")
def get_llm_config(user: Dict = Depends(get_current_user)):
    return {
        "api_key": LLM_API_KEY[:8] + "***" if LLM_API_KEY else "",
        "base_url": LLM_BASE_URL,
        "model": LLM_MODEL
    }


@app.put("/api/config/llm")
def update_llm_config(req: LLMConfigUpdate, user: Dict = Depends(require_super_admin)):
    import backend.config as cfg
    if req.api_key:
        cfg.LLM_API_KEY = req.api_key
    if req.base_url:
        cfg.LLM_BASE_URL = req.base_url
    if req.model:
        cfg.LLM_MODEL = req.model
    return {"success": True, "message": "LLM 配置已更新（运行时生效，重启后需修改 config.py）"}


# ── 日志接口 ──

@app.get("/api/logs")
def get_logs(log_type: str = "backend", lines: int = 200, user: Dict = Depends(get_current_user)):
    log_files = {
        "backend": LOG_DIR / "backend.log",
        "frontend": LOG_DIR / "frontend.log",
        "admin": LOG_DIR / "admin.log",
    }
    log_file = log_files.get(log_type)
    if not log_file or not log_file.exists():
        return {"logs": [], "file": str(log_file)}
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return {"logs": recent_lines, "file": str(log_file), "total_lines": len(all_lines)}
    except Exception as e:
        return {"logs": [], "error": str(e)}


# ── 用户管理接口 ──

@app.get("/api/users")
def list_users(user: Dict = Depends(get_current_user)):
    """返回按角色分组的名单：管理员名单 + 普通用户名单。

    兼容旧字段：仍保留扁平 users 字段（=全部账号）。
    """
    grouped = get_users_grouped()
    return {
        "admins": grouped["admins"],
        "users": grouped["users"],
        "admin_count": grouped["admin_count"],
        "user_count": grouped["user_count"],
        "all": get_all_users(),
    }


@app.delete("/api/users/{user_id}")
def remove_user(user_id: int, user: Dict = Depends(require_super_admin)):
    if user_id == user.get("id"):
        raise HTTPException(status_code=400, detail="不能删除自己")
    if not delete_user(user_id):
        raise HTTPException(status_code=400, detail="无法删除该用户（可能是超级管理员）")
    return {"success": True}


# ── 前端页面 ──

@app.get("/", response_class=HTMLResponse)
def admin_page():
    html_path = PROJECT_ROOT / "static" / "admin.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Admin panel HTML not found. Please check static/admin.html</h1>"


# ── 端口检测 ──

def find_available_port(start_port: int = ADMIN_PORT_START, max_tries: int = 10) -> int:
    for i in range(max_tries):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"无法找到可用端口 ({start_port}-{start_port + max_tries - 1})")


# ── 启动入口 ──

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    _init_admin_db()

    port = find_available_port()
    print("=" * 50)
    print("  HR Agent 开发者管理面板")
    print("=" * 50)
    print(f"  访问地址: http://localhost:{port}")
    print(f"  超级管理员: admin / admin")
    print(f"  简历目录: {RESUME_DATA_DIR}")
    print("=" * 50)

    # 启动简历扫描后台线程
    from backend.resume_scanner import start_scanner_thread
    start_scanner_thread()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
