#!/usr/bin/env python3
"""AIBP 智能人力助手 - HTTP 服务入口

接口：POST /chat
参数：session_id, message_id, emp_id, query

无状态设计：
  - 每次请求按需创建 Agent，处理完即销毁（不维护全局 Agent 实例）
  - 对话历史 + 挂起的交互状态，全部保存在 SessionStore（内存模拟 Redis）
  - 后期切换真实 Redis 只需替换 core/session_store.py 底部的 session_store 实例

交互式多轮说明：
  Agent 遇到需要用户确认时，response.interaction != null，
  此时已将"待执行上下文"序列化存入 SessionStore；
  下次调用时，服务从 SessionStore 恢复上下文直接进入执行阶段，跳过 LLM 意图解析。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logging_config import setup_logging
setup_logging(log_dir="logs/http_server")

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.session_store import SESSION_TTL, session_key, session_store
from agents.main_agent import run_query
from backend.skills.skill_registry import register_all_skills

# ── 注册所有 Skills ───────────────────────────────────────────────────────────
register_all_skills()

# ── 启动简历自动扫描后台线程 ──────────────────────────────────────────────────
from backend.resume_scanner import start_scanner_thread, ensure_resume_data_dir
ensure_resume_data_dir()
start_scanner_thread()

# ── FastAPI 应用 ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AIBP 智能人力助手",
    description="多Agent协同的智能HR问答服务（无状态版）",
    version="2.0.0",
)


# ── 请求 / 响应模型 ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message_id: str
    emp_id: str
    query: str


class StepInfo(BaseModel):
    stage: int
    title: str
    detail: str = ""


class InteractionInfo(BaseModel):
    interaction_id: str
    source: str
    stage: int
    interaction_type: str
    prompt: str
    options: List[Any] = []
    default: Any = None


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    emp_id: str
    answer: str = ""
    suggestions: List[str] = []
    sources: List[str] = []
    steps: List[StepInfo] = []
    interaction: Optional[InteractionInfo] = None


# ── 核心接口 ──────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    统一问答接口（无状态版）。

    正常流程：
        1. 验证 emp_id → 从 SessionStore 加载对话历史
        2. 调用 run_query() 处理（无状态，函数式）
        3. 把新的对话历史写回 SessionStore
        4. 返回结果

    交互续接流程（Agent 需要用户确认）：
        1. 首次请求返回 interaction != null，同时将待执行上下文序列化存入 SessionStore
        2. 调用方向用户展示 prompt，等待输入
        3. 再次调用 /chat，query 填用户回答，session_id 保持不变
        4. 服务从 SessionStore 读取挂起上下文，跳过意图解析直接执行
    """
    emp_id = req.emp_id
    session_id = req.session_id
    message_id = req.message_id

    logger.info(f"[/chat] session={session_id} message={message_id} emp={emp_id} query={req.query!r}")

    # ── 从 SessionStore 加载 session 状态 ────────────────────────────────────
    skey = session_key(session_id)
    session_data: dict = session_store.get(skey) or {
        "history": [],
        "pending_interaction": None,
    }

    # ── 调用无状态处理函数 ────────────────────────────────────────────────────
    result = run_query(
        session_id=session_id,
        message_id=message_id,
        query=req.query,
        emp_id=emp_id,
        history=session_data["history"],
        pending_interaction=session_data.get("pending_interaction"),
    )

    # ── 更新 SessionStore ─────────────────────────────────────────────────────
    session_data["history"] = result["history"]
    session_data["pending_interaction"] = result.get("pending_interaction")
    session_store.set(skey, session_data, ttl=SESSION_TTL)

    # ── 构造响应 ──────────────────────────────────────────────────────────────
    steps = [StepInfo(**s) for s in result.get("steps", [])]

    if result.get("pending_interaction"):
        pi = result["pending_interaction"]
        return ChatResponse(
            session_id=session_id,
            message_id=req.message_id,
            emp_id=emp_id,
            steps=steps,
            interaction=InteractionInfo(
                interaction_id=pi["interaction_id"],
                source=pi["source"],
                stage=pi["stage"],
                interaction_type=pi["interaction_type"],
                prompt=pi["prompt"],
                options=pi.get("options", []),
                default=pi.get("default"),
            ),
        )

    return ChatResponse(
        session_id=session_id,
        message_id=req.message_id,
        emp_id=emp_id,
        answer=result.get("answer", "⚠️ 处理异常，请稍后重试。"),
        suggestions=result.get("suggestions", []),
        sources=result.get("sources", []),
        steps=steps,
    )


# ── 用户反馈接口 ───────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str = ""
    rating: int  # 1=满意(点赞), 0=不满意(点踩)
    query: str = ""  # 可选：关联的原始查询


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """接收用户对回答质量的反馈（点赞/点踩），驱动模型权重和调度阈值动态调整。

    流程：
      1. 将反馈写入 matching_history（持久化）
      2. 通知 DynamicScheduler 进行阈值调整（用户满意度维度）
      3. 累积足够反馈后触发 CatBoost 特征权重更新
    """
    from backend.database.models import hr_db
    from backend.models.catboost_matcher import catboost_matcher
    from backend.harness.dynamic_scheduler import dynamic_scheduler

    logger.info(f"[/api/feedback] session={req.session_id} rating={req.rating}")

    # 1) 持久化反馈到 DB
    history_id = None
    try:
        history_id = hr_db.record_feedback_standalone(
            session_id=req.session_id,
            message_id=req.message_id,
            feedback=req.rating,
            query=req.query,
        )
    except Exception as e:
        logger.warning(f"Failed to persist feedback: {e}")

    # 2) 通知 DynamicScheduler（用户满意度维度）
    dynamic_scheduler.record_user_feedback(satisfied=(req.rating >= 1))

    # 3) 累积反馈后触发特征权重调整
    recent = hr_db.get_recent_feedback(50)
    if len(recent) >= 10:
        positive = sum(1 for r in recent if r.get("feedback", 0) >= 1)
        negative = len(recent) - positive
        satisfaction_rate = positive / len(recent) if recent else 0.5
        # 当满意率偏低时增加多样性权重；偏高时强化精确匹配权重
        catboost_matcher.update_weights({
            "satisfaction_rate": satisfaction_rate,
            "positive_count": positive,
            "negative_count": negative,
            "adjustment": "increase_diversity" if satisfaction_rate < 0.6 else "increase_precision",
        })

    return {
        "status": "ok",
        "history_id": history_id,
        "message": "反馈已记录，系统将据此优化后续推荐质量",
    }


# ── 辅助接口 ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.delete("/session/{session_id}")
def clear_session(session_id: str) -> Dict[str, str]:
    """清除指定 session 的状态（对话历史 + 挂起交互）"""
    removed = session_store.delete(session_key(session_id))
    return {"status": "cleared" if removed else "not_found", "session_id": session_id}


# ── 启动入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("http_server:app", host="0.0.0.0", port=8003, reload=False)
