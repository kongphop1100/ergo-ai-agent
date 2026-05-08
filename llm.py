"""OpenRouter chat-completions wrapper.

Same pattern as backend/main.py:_call_openrouter, but with two extensions:
- accepts a `tools` list (OpenAI function-calling format — OpenRouter passes through)
- returns the FULL response (not just the content string) so the caller can read
  tool_calls and finish_reason.

Uses sync httpx because Streamlit runs synchronously per request.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

_ENV_LOADED = False


def _load_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    here = Path(__file__).parent
    candidates = [here / ".env", here.parent / "backend" / ".env", here.parent / ".env"]
    for c in candidates:
        if c.exists():
            load_dotenv(dotenv_path=c)
            break
    else:
        load_dotenv()  # cwd fallback
    _ENV_LOADED = True


def get_api_key() -> str:
    _load_env_once()
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def get_default_model() -> str:
    _load_env_once()
    return os.getenv(
        "OPENROUTER_TEXT_MODEL",
        os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it"),
    ).strip()


def get_default_vision_model() -> str:
    _load_env_once()
    return os.getenv(
        "OPENROUTER_VISION_MODEL",
        get_default_model(),
    ).strip()


def get_base_url() -> str:
    _load_env_once()
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")


def call_openrouter(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict | None = None,
    response_format: dict | None = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> dict:
    """Returns the full assistant message dict (with tool_calls if any)."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY missing. Set it in streamlit_app/.env or backend/.env."
        )
    payload: dict[str, Any] = {
        "model": model or get_default_model(),
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ERGO AI Office Syndrome Agent",
    }
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        resp = client.post(f"{get_base_url()}/chat/completions", headers=headers, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    try:
        return body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {body}") from exc


def parse_model_json(content: str) -> dict | None:
    """Strict JSON first, then salvage the first {...} block."""
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
