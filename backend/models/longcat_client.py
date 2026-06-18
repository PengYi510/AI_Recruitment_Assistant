"""LLM客户端 - 保留原有LongCat API调用方式"""
import json, logging, time
from typing import Any, Optional, List, Dict
from openai import OpenAI, RateLimitError
from backend.config import (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_RATE_LIMIT_RETRIES, LLM_RATE_LIMIT_WAIT, PROJECT_ROOT)

logger = logging.getLogger(__name__)

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
    return resp.choices[0].message
