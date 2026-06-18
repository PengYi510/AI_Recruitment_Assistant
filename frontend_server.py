"""
HR Agent Web API 服务
提供前后端分离的 RESTful API 接口，支持 SSE 流式推送

接口列表:
  POST /api/auth/login                      - 用户登录
  POST /api/auth/register                   - 用户注册
  GET  /api/auth/me                         - 获取当前用户信息
  POST /api/sessions                        - 创建新会话
  GET  /api/sessions                        - 获取所有会话列表
  DELETE /api/sessions/<session_id>         - 删除会话
  POST /api/chat                            - 发送消息（SSE 流式返回）
  GET  /api/sessions/<session_id>/history   - 获取会话历史记录
  GET  /api/health                          - 健康检查

运行方式:
    python frontend_server.py
"""

import json
import logging
import os
import sys
import threading
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
from functools import wraps

from core.logging_config import setup_logging
setup_logging(log_dir="logs/frontend_server")

import requests as _backend_requests

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS

from session_db import session_db

# ── 后端 http_server 地址 ─────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8003")

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Flask 应用初始化
# ──────────────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.config["JSON_AS_ASCII"] = False
app.secret_key = os.environ.get("SECRET_KEY", "hr-agent-secret-key")
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "*"}})


# ──────────────────────────────────────────────────────────────────────
# Token 管理 & 认证
# ──────────────────────────────────────────────────────────────────────

# 简易 Token 存储: token -> {"username": ..., "expires": datetime}
_token_store: Dict[str, dict] = {}
_token_lock = threading.Lock()
TOKEN_EXPIRE_HOURS = 24


def _generate_token(username: str) -> str:
    """为用户生成一个随机 Token 并存储"""
    token = secrets.token_hex(32)
    with _token_lock:
        _token_store[token] = {
            "username": username,
            "expires": datetime.now() + timedelta(hours=TOKEN_EXPIRE_HOURS),
        }
    return token


def _validate_token(token: str) -> Optional[str]:
    """验证 Token，返回用户名或 None"""
    with _token_lock:
        entry = _token_store.get(token)
        if entry is None:
            return None
        if datetime.now() > entry["expires"]:
            del _token_store[token]
            return None
        return entry["username"]


def login_required(f):
    """认证装饰器：需要有效 Token 才能访问"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        if not token:
            token = request.args.get("token")
        if not token:
            return jsonify({"error": "未登录，请先登录"}), 401
        username = _validate_token(token)
        if username is None:
            return jsonify({"error": "登录已过期，请重新登录"}), 401
        request.current_user = username
        return f(*args, **kwargs)
    return decorated

# ──────────────────────────────────────────────────────────────────────
# 会话管理
# ──────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    内存会话存储
    管理会话元信息（session_id / 标题 / 消息计数）。
    对话处理完全由后端 http_server 负责，此处不再持有 HRAgent 实例。
    """

    def __init__(self):
        self._meta: Dict[str, dict] = {}               # session_id -> 元信息
        self._lock = threading.Lock()

    def restore_from_database(self):
        """从数据库恢复会话元信息。在服务启动时调用。"""
        logger.info("从数据库恢复会话元信息...")
        try:
            with self._lock:
                self._meta.clear()
                all_sessions = session_db.get_all_sessions()
                for session_meta in all_sessions:
                    sid = session_meta.get("session_id")
                    if sid:
                        self._meta[sid] = session_meta
                logger.info(f"恢复完成: 会话元信息 {len(self._meta)} 个")
        except Exception as e:
            logger.error(f"从数据库恢复会话失败: {e}", exc_info=True)

    def create_session(self, user_id: Optional[str] = None) -> dict:
        """创建新会话，返回会话元信息"""
        session_id = str(uuid.uuid4())[:8]
        with self._lock:
            meta = {
                "session_id": session_id,
                "user_id": user_id,
                "user_mis": user_id or "",  # 与数据库字段对齐
                "title": "新对话",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "message_count": 0,
            }
            self._meta[session_id] = meta
            # 保存到数据库，实现持久化
            session_db.save_session(meta)
        logger.info(f"创建新会话: {session_id} (user={user_id})")
        return meta

    def get_meta(self, session_id: str) -> Optional[dict]:
        """获取会话元信息"""
        meta = self._meta.get(session_id)
        if meta is None:
            # 内存中没有，尝试从数据库恢复
            meta = session_db.get_session(session_id)
            if meta is None:
                return None
            with self._lock:
                self._meta[session_id] = meta
        return meta

    def list_sessions(self) -> list:
        """返回所有会话列表（按更新时间倒序）"""
        db_sessions = session_db.get_all_sessions()
        with self._lock:
            for s in db_sessions:
                sid = s.get("session_id")
                if sid and sid not in self._meta:
                    self._meta[sid] = s
        return db_sessions

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话"""
        with self._lock:
            meta = self._meta.get(session_id)
            if meta is None:
                meta = session_db.get_session(session_id)
                if meta is None:
                    return False
            self._meta.pop(session_id, None)
            session_db.delete_session(session_id)
        logger.info(f"删除会话: {session_id}")
        return True

    def update_session_title(self, session_id: str, title: str):
        """更新会话标题"""
        with self._lock:
            if session_id in self._meta:
                self._meta[session_id]["title"] = title
                self._meta[session_id]["updated_at"] = datetime.now().isoformat()
                session_db.update_session_title(session_id, title)

    def increment_message_count(self, session_id: str):
        """递增消息计数并更新时间戳"""
        with self._lock:
            if session_id in self._meta:
                self._meta[session_id]["message_count"] += 1
                self._meta[session_id]["updated_at"] = datetime.now().isoformat()
                session_db.increment_message_count(session_id)


# 全局会话存储单例
session_store = SessionStore()


# ──────────────────────────────────────────────────────────────────────
# SSE 工具函数
# ──────────────────────────────────────────────────────────────────────

def sse_event(event: str, data: dict) -> str:
    """构造一条 SSE 消息"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    """SSE 结束标记"""
    return "event: done\ndata: {}\n\n"


def sse_error(message: str) -> str:
    """SSE 错误消息"""
    return f"event: error\ndata: {json.dumps({'message': message}, ensure_ascii=False)}\n\n"


def sse_suggestions(suggestions: list) -> str:
    """SSE 追问建议事件"""
    return f"event: suggestions\ndata: {json.dumps({'suggestions': suggestions}, ensure_ascii=False)}\n\n"


# ──────────────────────────────────────────────────────────────────────
# 认证 API 路由
# ──────────────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    """
    用户注册

    Request Body:
        { "username": "xxx", "password": "xxx", "nickname": "xxx" }

    Response:
        { "success": true, "user": {...}, "token": "..." }
    """
    body = request.get_json(silent=True) or {}
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()
    nickname = body.get("nickname", "").strip()

    if not username or not password:
        return jsonify({"success": False, "error": "用户名和密码不能为空"}), 400
    if len(username) < 2 or len(username) > 20:
        return jsonify({"success": False, "error": "用户名长度需在2-20个字符之间"}), 400
    if len(password) < 4 or len(password) > 32:
        return jsonify({"success": False, "error": "密码长度需在4-32个字符之间"}), 400

    result = session_db.register_user(username, password, nickname)
    if result["success"]:
        token = _generate_token(username)
        return jsonify({"success": True, "user": result["user"], "token": token}), 201
    else:
        return jsonify(result), 409


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    """
    用户登录

    Request Body:
        { "username": "xxx", "password": "xxx" }

    Response:
        { "success": true, "user": {...}, "token": "..." }
    """
    body = request.get_json(silent=True) or {}
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()

    if not username or not password:
        return jsonify({"success": False, "error": "用户名和密码不能为空"}), 400

    result = session_db.authenticate_user(username, password)
    if result["success"]:
        token = _generate_token(username)
        return jsonify({"success": True, "user": result["user"], "token": token})
    else:
        return jsonify(result), 401


@app.route("/api/auth/me", methods=["GET"])
@login_required
def api_me():
    """
    获取当前登录用户信息

    Headers:
        Authorization: Bearer <token>

    Response:
        { "success": true, "user": {...} }
    """
    user = session_db.get_user_by_username(request.current_user)
    if user:
        return jsonify({"success": True, "user": user})
    return jsonify({"success": False, "error": "用户不存在"}), 404


@app.route("/api/auth/logout", methods=["POST"])
@login_required
def api_logout():
    """用户登出，使 Token 失效"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        with _token_lock:
            _token_store.pop(token, None)
    return jsonify({"success": True})


# ──────────────────────────────────────────────────────────────────────
# 前端静态文件路由
# ──────────────────────────────────────────────────────────────────────

@app.route('/')
def root():
    """根路由 - 跳转到登录页"""
    return send_from_directory(STATIC_DIR, "login.html")


@app.route("/login")
def serve_login():
    """登录页面"""
    return send_from_directory(STATIC_DIR, "login.html")


@app.route("/chat")
def serve_frontend():
    """提供前端 IM 页面"""
    return send_from_directory(STATIC_DIR, "index2.html")


@app.route("/data/shap/<path:filename>")
def serve_shap_chart(filename):
    """提供 SHAP 可视化图表（瀑布图、全局重要性图等）
    
    如果请求的瀑布图不存在，自动为该候选人生成一张。
    """
    shap_dir = os.path.join(os.path.dirname(__file__), "data", "shap")
    full_path = os.path.join(shap_dir, filename)

    # 如果文件不存在且是 waterfall.png 请求，尝试动态生成
    if not os.path.exists(full_path) and filename.endswith("waterfall.png"):
        try:
            # 从路径中提取候选人ID（格式: {candidate_id}/waterfall.png）
            candidate_id = int(filename.split("/")[0])
            logger.info(f"[SHAP] 瀑布图不存在，动态生成: candidate {candidate_id}")
            _generate_shap_for_candidate(candidate_id)
        except (ValueError, IndexError):
            pass  # 路径格式不对，跳过

    # 如果请求全局重要性图且不存在，自动生成
    if not os.path.exists(full_path) and filename == "global_importance.png":
        try:
            logger.info("[SHAP] 全局特征重要性图不存在，动态生成")
            _generate_global_importance()
        except Exception as e:
            logger.error(f"[SHAP] 全局重要性图生成失败: {e}")

    return send_from_directory(shap_dir, filename)


def _generate_global_importance():
    """生成全局特征重要性柱状图（data/shap/global_importance.png）"""
    import asyncio
    try:
        from backend.skills.shap_explainer_skill import SHAPExplainerSkill
        import numpy as np

        skill = SHAPExplainerSkill()
        # 使用平均特征值生成全局图
        features_list = [0.5] * 12
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(skill.execute({
                'candidate_id': None,
                'features': features_list,
                'match_score': 0.5,
                'level': 'global',
                'query': '综合匹配评估',
            }))
        finally:
            loop.close()
        logger.info("[SHAP] 全局特征重要性图生成完成")
    except Exception as e:
        logger.error(f"[SHAP] 全局特征重要性图生成失败: {e}")


def _generate_shap_for_candidate(candidate_id: int):
    """为指定候选人动态生成 SHAP 瀑布图（当图不存在时的 fallback）"""
    import asyncio
    try:
        from backend.skills.shap_explainer_skill import SHAPExplainerSkill
        from backend.models.catboost_matcher import catboost_matcher
        from backend.database.models import hr_db
        import numpy as np

        skill = SHAPExplainerSkill()
        # 获取候选人数据以提取结构化特征
        candidate = hr_db.get_candidate(candidate_id)
        if candidate:
            # 使用空 JD 提取基础特征
            features = catboost_matcher.extract_structured_features({}, candidate)
            features_list = features.tolist()
        else:
            # 候选人不在数据库中，使用基于ID的确定性随机特征
            np.random.seed(candidate_id)
            features_list = np.random.rand(12).tolist()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(skill.execute({
                'candidate_id': candidate_id,
                'features': features_list,
                'match_score': 0.7,
                'level': 'individual',
                'query': '综合匹配评估',
            }))
        finally:
            loop.close()
        logger.info(f"[SHAP] 动态生成完成: candidate {candidate_id}")
    except Exception as e:
        logger.error(f"[SHAP] 动态生成失败: candidate {candidate_id}, error: {e}")


@app.route("/api/shap/explain/<int:candidate_id>", methods=["GET"])
def get_shap_explanation(candidate_id: int):
    """获取候选人的 SHAP 四层解释数据（交互解释 + 自然语言解释）

    返回 JSON:
    {
        "interaction_explanation": {"top_interactions": [...]},
        "nlp_explanation": {"explanation": "...", ...},
        "global_explanation": {"feature_importance": {...}}
    }
    """
    import asyncio
    try:
        from backend.skills.shap_explainer_skill import SHAPExplainerSkill
        from backend.models.catboost_matcher import catboost_matcher
        from backend.database.models import hr_db
        import numpy as np

        skill = SHAPExplainerSkill()
        candidate = hr_db.get_candidate(candidate_id)
        if candidate:
            features = catboost_matcher.extract_structured_features({}, candidate)
            features_list = features.tolist()
        else:
            np.random.seed(candidate_id)
            features_list = np.random.rand(12).tolist()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(skill.execute({
                'candidate_id': candidate_id,
                'features': features_list,
                'match_score': 0.7,
                'level': 'all',
                'query': '综合匹配评估',
            }))
        finally:
            loop.close()

        # 提取需要的字段返回给前端
        response = {
            "interaction_explanation": result.get("interaction_explanation", {}),
            "nlp_explanation": result.get("nlp_explanation", {}),
            "global_explanation": result.get("global_explanation", {}),
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"[SHAP] 解释数据生成失败: candidate {candidate_id}, error: {e}")
        return jsonify({"error": str(e)}), 500


def _estimate_age(candidate: dict) -> int:
    """从教育经历推算候选人年龄

    策略：找到最早的本科/专科入学时间，假设入学时18岁，推算当前年龄。
    如果没有教育经历，则用 22 + work_years 估算。
    """
    import re
    from datetime import date

    current_year = date.today().year
    edu_history = candidate.get("education_history", [])

    if edu_history:
        earliest_start_year = None
        for edu in edu_history:
            if not isinstance(edu, dict):
                continue
            start_date = edu.get("start_date", "")
            if not start_date:
                continue
            # 提取年份（支持 "2018.09"、"2018-09"、"2018" 等格式）
            match = re.match(r"(\d{4})", str(start_date))
            if match:
                year = int(match.group(1))
                if earliest_start_year is None or year < earliest_start_year:
                    earliest_start_year = year

        if earliest_start_year:
            # 假设最早入学（本科/专科）时18岁
            return current_year - earliest_start_year + 18

    # 兜底：用 work_years 估算
    try:
        work_years = float(candidate.get("work_years") or 0)
        return int(22 + work_years)
    except (ValueError, TypeError):
        return 22


@app.route("/api/candidate/<int:candidate_id>", methods=["GET"])
def get_candidate_detail(candidate_id: int):
    """获取候选人详细信息（含教育经历、技能、工作经历等）用于前端简历面板展示

    前端传入的 candidate_id 实际上是候选人名字中的数字（如"候选人_2186"中的 2186），
    而非数据库自增 ID。因此优先按名字匹配，找不到时再按 ID 查询。
    """
    try:
        from backend.database.models import hr_db

        # 优先按名字中的数字匹配（前端传入的是名字编号）
        candidate = hr_db.get_candidate_by_name_number(candidate_id)

        # 兜底：按数据库 ID 查询
        if not candidate:
            candidate = hr_db.get_candidate(candidate_id)

        if not candidate:
            return jsonify({"error": "候选人不存在"}), 404

        # 补全缺失的 age 字段：优先从教育经历最早入学时间推算
        if not candidate.get("age"):
            candidate["age"] = _estimate_age(candidate)

        return jsonify(candidate)
    except Exception as e:
        logger.error(f"获取候选人详情失败: {e}")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────────────
# API 路由
# ──────────────────────────────────────────────────────────────────────

@app.route("/api/feedback", methods=["POST"])
@login_required
def submit_feedback():
    """接收用户反馈（点赞/点踩），驱动模型权重和调度阈值动态调整。

    Request Body:
        { "session_id": "...", "message_id": "...", "rating": 1 或 0 }
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    message_id = data.get("message_id", "")
    rating = data.get("rating")

    if rating is None or rating not in (0, 1):
        return jsonify({"error": "rating 必须为 0 或 1"}), 400

    from backend.database.models import hr_db
    from backend.models.catboost_matcher import catboost_matcher
    from backend.harness.dynamic_scheduler import dynamic_scheduler

    logger.info(f"[/api/feedback] session={session_id} rating={rating}")

    # 1) 持久化反馈到 DB
    history_id = None
    try:
        history_id = hr_db.record_feedback_standalone(
            session_id=session_id,
            message_id=message_id,
            feedback=rating,
            query=data.get("query", ""),
        )
    except Exception as e:
        logger.warning(f"Failed to persist feedback: {e}")

    # 2) 通知 DynamicScheduler（用户满意度维度）
    dynamic_scheduler.record_user_feedback(satisfied=(rating >= 1))

    # 3) 累积反馈后触发特征权重调整
    recent = hr_db.get_recent_feedback(50)
    if len(recent) >= 10:
        positive = sum(1 for r in recent if r.get("feedback", 0) >= 1)
        negative = len(recent) - positive
        satisfaction_rate = positive / len(recent) if recent else 0.5
        catboost_matcher.update_weights({
            "satisfaction_rate": satisfaction_rate,
            "positive_count": positive,
            "negative_count": negative,
            "adjustment": "increase_diversity" if satisfaction_rate < 0.6 else "increase_precision",
        })

    return jsonify({
        "status": "ok",
        "history_id": history_id,
        "message": "反馈已记录，系统将据此优化后续推荐质量",
    })


@app.route("/api/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/sessions", methods=["POST"])
@login_required
def create_session():
    """
    创建新会话
    
    Request Body (可选):
        { "user_id": "xxx" }
    
    Response:
        { "session_id": "...", "title": "新对话", "created_at": "..." }
    """
    body = request.get_json(silent=True) or {}
    user_id = request.current_user  # 使用当前登录用户
    meta = session_store.create_session(user_id=user_id)
    return jsonify(meta), 201


@app.route("/api/sessions", methods=["GET"])
@login_required
def list_sessions():
    """
    获取当前用户的会话列表
    
    Response:
        { "sessions": [...] }
    """
    username = request.current_user
    # 按用户过滤会话
    all_sessions = session_store.list_sessions()
    user_sessions = [s for s in all_sessions if s.get("user_mis") == username or s.get("user_id") == username]
    return jsonify({"sessions": user_sessions})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """
    删除指定会话
    
    Response:
        { "success": true }
    """
    success = session_store.delete_session(session_id)
    if not success:
        return jsonify({"error": "会话不存在"}), 404
    return jsonify({"success": True})


@app.route("/api/sessions/<session_id>/history", methods=["GET"])
def get_history(session_id: str):
    """
    获取会话历史消息
    
    Response:
        { "session_id": "...", "messages": [...] }
    """
    meta = session_store.get_meta(session_id)
    if meta is None:
        return jsonify({"error": "会话不存在"}), 404

    db_messages = session_db.get_messages(session_id)
    messages = [
        {
            "id": f"{session_id}_{i}",
            "role": m.get("role", "assistant"),
            "content": m.get("content", ""),
        }
        for i, m in enumerate(db_messages)
    ]
    return jsonify({"session_id": session_id, "messages": messages})


@app.route("/api/short_memory", methods=["GET"])
def get_short_memory():
    """
    获取近期短期记忆内容（今天 + 昨天的 md 文件合并）

    Query Params:
        reload=true   强制重新读取文件（默认使用缓存）
        user_mis=xxx  指定用户MIS（可选）

    Response:
        { "content": "...", "has_memory": true/false }
    """
    from short_memory import get_short_memory_context
    user_mis = request.args.get("user_mis")
    force_reload = request.args.get("reload", "").lower() == "true"
    content = get_short_memory_context(force_reload=force_reload, user_mis=user_mis)
    return jsonify({
        "content": content,
        "has_memory": bool(content.strip()),
        "length": len(content),
    })


@app.route("/api/long_term_memory", methods=["GET"])
def get_long_term_memory():
    """
    获取用户的双层长期记忆

    Query Params:
        user_id=xxx  用户标识（MIS号）

    Response:
        {
            "persistent": [...],   // 显式长期记忆列表
            "adaptive": [...],     // 自适应记忆列表
            "prompt_text": "..."   // 合并后的 prompt 注入文本
        }
    """
    from backend.memory import memory_loader as _ml
    user_id = request.args.get("user_id", "default")
    persistent = _ml.persistent.get_all(user_id)
    adaptive = _ml.adaptive.get_active(user_id, limit=20)
    prompt_text = _ml.load_memory_context(user_id)
    return jsonify({
        "persistent": persistent,
        "adaptive": adaptive,
        "prompt_text": prompt_text,
        "has_memory": bool(persistent or adaptive),
    })


@app.route("/api/long_term_memory", methods=["POST"])
def save_long_term_memory():
    """
    手动保存一条长期记忆（管理接口）

    Request Body:
        {
            "user_id": "xxx",
            "content": "QS前50等于985",
            "category": "rule",       // rule/preference/definition
            "layer": "persistent"     // persistent/adaptive
        }
    """
    from backend.memory import memory_loader as _ml
    data = request.get_json(force=True)
    user_id = data.get("user_id", "default")
    content = data.get("content", "").strip()
    category = data.get("category", "observation")
    query_context = data.get("query_context", "")

    if not content:
        return jsonify({"error": "content 不能为空"}), 400

    # 统一通过 save_memory 入口，由重要性评分自动决定分层
    result = _ml.save_memory(
        user_id=user_id,
        content=content,
        query_context=query_context,
        category=category,
    )

    return jsonify({"success": True, **result})


@app.route("/api/long_term_memory/<int:memory_id>", methods=["DELETE"])
def delete_long_term_memory(memory_id):
    """
    删除（停用）一条长期记忆

    Query Params:
        layer=persistent  指定层级（persistent/adaptive）
    """
    from backend.memory import persistent_memory, adaptive_memory
    layer = request.args.get("layer", "persistent")
    if layer == "persistent":
        persistent_memory.deactivate(memory_id)
    else:
        # adaptive_memory 是 AdaptiveMemoryStore 实例，直接通过 SQL 停用
        with adaptive_memory._get_conn() as conn:
            conn.execute(
                "UPDATE adaptive_memory SET is_active = 0 WHERE id = ?",
                (memory_id,)
            )
    return jsonify({"success": True, "deactivated_id": memory_id})


@app.route("/api/long_term_memory/consolidate", methods=["POST"])
def consolidate_memory():
    """
    触发自适应记忆的定期总结与清理

    Request Body:
        { "user_id": "xxx" }
    """
    from backend.memory import memory_loader as _ml
    data = request.get_json(force=True)
    user_id = data.get("user_id", "default")
    stats = _ml.run_consolidation(user_id)
    return jsonify({"success": True, "stats": stats})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    发送消息并以 SSE 流的形式接收回复。

    调用链：前端 → frontend_server(/api/chat) → http_server(/chat) → Agent

    Request Body:
        {
            "session_id": "abc12345",   // 必填，已创建的会话 ID
            "message": "帮我找一个..."  // 必填，用户消息内容
        }

    SSE Events:
        event: thinking    -> { "status": "thinking" }         Agent 开始处理
        event: message     -> { "content": "..." }             完整回复内容
        event: meta        -> { "session_id":"...", "title":"..." }  元信息更新
        event: suggestions -> { "suggestions": [...] }         追问建议（有则推送）
        event: done        -> {}                                处理完成
        event: error       -> { "message": "..." }             处理出错

    Content-Type: text/event-stream
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "请求体不能为空"}), 400

    session_id = body.get("session_id", "").strip()
    user_message = body.get("message", "").strip()

    if not session_id:
        return jsonify({"error": "session_id 不能为空"}), 400
    if not user_message:
        return jsonify({"error": "message 不能为空"}), 400

    meta = session_store.get_meta(session_id)
    if meta is None:
        return jsonify({"error": f"会话 {session_id} 不存在"}), 404

    emp_id = body.get("emp_id", "")
    message_id = f"msg-{uuid.uuid4().hex[:8]}"

    def generate():
        # 1. 通知前端：开始处理
        yield sse_event("thinking", {"status": "thinking", "text": "正在理解您的意图"})

        # 2. 调用后端 http_server /chat 接口
        try:
            resp = _backend_requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "session_id": session_id,
                    "message_id": message_id,
                    "emp_id": emp_id,
                    "query": user_message,
                },
                timeout=180,
            )
        except _backend_requests.exceptions.Timeout:
            logger.error(f"调用后端超时: session={session_id}")
            yield sse_error("请求处理超时，请稍后重试")
            yield sse_done()
            return
        except Exception as e:
            logger.error(f"调用后端异常: {e}", exc_info=True)
            yield sse_error(f"服务内部错误: {str(e)}")
            yield sse_done()
            return

        if resp.status_code != 200:
            detail = ""
            try:
                detail = resp.json().get("detail", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            logger.error(f"后端返回错误: status={resp.status_code}, detail={detail}")
            yield sse_error(f"处理失败（{resp.status_code}）: {detail}")
            yield sse_done()
            return

        data = resp.json()
        answer = data.get("answer", "")
        suggestions = data.get("suggestions", [])
        interaction = data.get("interaction")

        # 3. 将用户消息和 AI 回复持久化到数据库
        session_db.save_messages(session_id, "user", user_message)
        if answer:
            session_db.save_messages(session_id, "assistant", answer)

        # 4. 更新会话元信息（消息计数 + 标题）
        session_store.increment_message_count(session_id)
        current_meta = session_store.get_meta(session_id)
        if current_meta and current_meta.get("message_count", 0) <= 1:
            title = user_message[:20] + ("..." if len(user_message) > 20 else "")
            session_store.update_session_title(session_id, title)
        meta_info = session_store.get_meta(session_id)

        # 5. 推送回复内容
        if answer:
            yield sse_event("message", {"content": answer})

        # 5b. 若后端要求用户交互，推送 interaction 事件
        if interaction:
            yield sse_event("interaction", {
                "interaction_id":   interaction.get("interaction_id", ""),
                "source":           interaction.get("source", ""),
                "interaction_type": interaction.get("interaction_type", "select"),
                "prompt":           interaction.get("prompt", ""),
                "options":          interaction.get("options", []),
                "default":          interaction.get("default"),
            })

        # 6. 推送元信息
        yield sse_event("meta", {
            "session_id": session_id,
            "title": meta_info.get("title", "新对话") if meta_info else "新对话",
            "message_count": meta_info.get("message_count", 0) if meta_info else 0,
        })

        # 7. 推送追问建议（有才推送）
        if suggestions:
            yield sse_suggestions(suggestions)

        yield sse_done()

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ──────────────────────────────────────────────────────────────────────
# 程序入口
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HR Agent Web API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=9033, help="监听端口（默认 9033）")
    parser.add_argument("--debug", action="store_true", help="启用 Debug 模式")
    args = parser.parse_args()

    # 服务启动时从数据库恢复会话元信息
    logger.info("服务初始化中...")
    session_store.restore_from_database()

    logger.info(f"启动 HR Agent API 服务: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)
