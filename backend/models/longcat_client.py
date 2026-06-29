"""LLM客户端 - 保留原有LongCat API调用方式"""
import json, logging, time, threading
from typing import Any, Optional, List, Dict
from openai import OpenAI, RateLimitError
from backend.config import (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_RATE_LIMIT_RETRIES, LLM_RATE_LIMIT_WAIT, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ── Token 用量追踪器（线程安全） ──────────────────────────────────────────────
class TokenUsageTracker:
    """线程本地的 Token 用量累加器。
    
    使用方式：
        tracker = get_token_tracker()
        tracker.reset()       # 请求开始时重置
        # ... 中间所有 LLM 调用会自动累加 ...
        usage = tracker.get() # 请求结束时获取累计值
    """
    def __init__(self):
        self._local = threading.local()

    def reset(self):
        self._local.prompt_tokens = 0
        self._local.completion_tokens = 0
        self._local.total_tokens = 0
        self._local.llm_calls = 0

    def add(self, usage):
        """累加一次 LLM 调用的 token 用量（usage 可以是 OpenAI Usage 对象或 dict）"""
        if usage is None:
            return
        if hasattr(usage, 'prompt_tokens'):
            pt = usage.prompt_tokens or 0
            ct = usage.completion_tokens or 0
            tt = usage.total_tokens or 0
        elif isinstance(usage, dict):
            pt = usage.get('prompt_tokens', 0) or 0
            ct = usage.get('completion_tokens', 0) or 0
            tt = usage.get('total_tokens', 0) or 0
        else:
            return
        self._local.prompt_tokens = getattr(self._local, 'prompt_tokens', 0) + pt
        self._local.completion_tokens = getattr(self._local, 'completion_tokens', 0) + ct
        self._local.total_tokens = getattr(self._local, 'total_tokens', 0) + tt
        self._local.llm_calls = getattr(self._local, 'llm_calls', 0) + 1

    def get(self) -> Dict[str, int]:
        return {
            "prompt_tokens": getattr(self._local, 'prompt_tokens', 0),
            "completion_tokens": getattr(self._local, 'completion_tokens', 0),
            "total_tokens": getattr(self._local, 'total_tokens', 0),
            "llm_calls": getattr(self._local, 'llm_calls', 0),
        }

_token_tracker = TokenUsageTracker()

def get_token_tracker() -> TokenUsageTracker:
    return _token_tracker

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
MODEL = LLM_MODEL
_PROMPT_DIR = PROJECT_ROOT / "prompt_library"

def load_prompt(agent_name: str, template_name: str) -> str:
    path = _PROMPT_DIR / agent_name / f"{template_name}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""

def _create_with_retry(**kwargs):
    for attempt in range(LLM_RATE_LIMIT_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt < LLM_RATE_LIMIT_RETRIES:
                time.sleep(LLM_RATE_LIMIT_WAIT)
            else:
                raise

def chat_completion(system: str, user: str, tools=None, temperature: float = 0.2, max_tokens: int = 4096):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs: Dict[str, Any] = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = _create_with_retry(**kwargs)
    # 累加 token 用量到全局追踪器
    if hasattr(resp, 'usage') and resp.usage:
        _token_tracker.add(resp.usage)
    return resp.choices[0].message

def chat_json(system: str, user: str, temperature: float = 0.1) -> dict:
    msg = chat_completion(system=system, user=user, temperature=temperature)
    text = (msg.content or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())

def chat_messages(messages: List[Dict[str, str]], temperature: float = 0.2, tools=None):
    kwargs: Dict[str, Any] = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = _create_with_retry(**kwargs)
    # 累加 token 用量到全局追踪器
    if hasattr(resp, 'usage') and resp.usage:
        _token_tracker.add(resp.usage)
    return resp.choices[0].message
