"""OpenAI-compatible API execution with resumable request-level caches."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from .data import SYSTEM_PROMPT, canonical_json, sha256_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_failure(message: str) -> str:
    lowered = message.lower()
    if any(
        term in lowered
        for term in (
            "insufficient_quota",
            "insufficient balance",
            "insufficient account balance",
            "credit balance",
        )
    ):
        return "INSUFFICIENT_BALANCE_OR_QUOTA"
    if any(term in lowered for term in ("rate limit", "tpm", "rpm", "too many requests")):
        return "RATE_LIMIT"
    if any(term in lowered for term in ("context length", "maximum context", "too many tokens")):
        return "CONTEXT_LIMIT"
    if any(term in lowered for term in ("unauthorized", "authentication", "api key", "401")):
        return "AUTHENTICATION"
    if any(term in lowered for term in ("timeout", "timed out")):
        return "TIMEOUT"
    return "OTHER_API_FAILURE"


class RollingRateLimiter:
    def __init__(self, tokens_per_minute: int | None, requests_per_minute: int | None):
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute
        self.events: deque[tuple[float, int]] = deque()
        self.lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        if not self.tokens_per_minute and not self.requests_per_minute:
            return
        if self.tokens_per_minute and tokens > self.tokens_per_minute:
            raise ValueError(
                f"One request estimates {tokens} tokens, exceeding the configured "
                f"TPM limit {self.tokens_per_minute}"
            )
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0][0] >= 60.0:
                    self.events.popleft()
                used_tokens = sum(value for _, value in self.events)
                token_ok = not self.tokens_per_minute or used_tokens + tokens <= self.tokens_per_minute
                request_ok = not self.requests_per_minute or len(self.events) < self.requests_per_minute
                if token_ok and request_ok:
                    self.events.append((now, tokens))
                    return
                wait_seconds = max(0.05, 60.0 - (now - self.events[0][0])) if self.events else 0.05
            time.sleep(min(wait_seconds, 5.0))


def validate_provider(name: str, provider: dict[str, Any]) -> None:
    required = ("base_url", "model", "key_env", "context_window_tokens", "max_output_tokens")
    missing = [key for key in required if not provider.get(key)]
    if missing:
        raise ValueError(f"Provider {name} is enabled but missing: {', '.join(missing)}")
    if provider.get("api_style") != "openai_chat_completions":
        raise ValueError(f"Unsupported API style for {name}: {provider.get('api_style')}")
    multiplier = float(provider.get("prompt_token_safety_multiplier", 1.0))
    if multiplier < 1.0:
        raise ValueError(
            f"Provider {name} prompt_token_safety_multiplier must be at least 1.0"
        )


def enabled_providers(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {
        name: provider
        for name, provider in config["providers"].items()
        if bool(provider.get("enabled"))
    }
    for name, provider in result.items():
        validate_provider(name, provider)
    return result


def common_max_input_tokens(config: dict[str, Any]) -> tuple[int, dict[str, int]]:
    providers = enabled_providers(config)
    if not providers:
        raise ValueError("No providers are enabled; fill provider model/context fields first")
    reserve = int(config["protocol"]["context_reserve_tokens"])
    limits = {
        name: math.floor(
            (
                int(provider["context_window_tokens"])
                - int(provider["max_output_tokens"])
                - reserve
            )
            / float(provider.get("prompt_token_safety_multiplier", 1.0))
        )
        for name, provider in providers.items()
    }
    if min(limits.values()) <= 0:
        raise ValueError(f"Provider context budget is not positive: {limits}")
    return min(limits.values()), limits


def build_request(
    provider: dict[str, Any], user_message: str, temperature: float
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": str(provider["model"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }
    if not provider.get("omit_temperature", False):
        request["temperature"] = float(temperature)
    parameter = str(provider.get("max_tokens_parameter", "max_tokens"))
    request[parameter] = int(provider["max_output_tokens"])
    if provider.get("json_object_mode", True):
        request["response_format"] = {"type": "json_object"}
    request.update(dict(provider.get("extra_request", {})))
    return request


def request_sha256(provider_name: str, provider: dict[str, Any], request: dict[str, Any]) -> str:
    attestation = {
        "provider": provider_name,
        "base_url": provider["base_url"],
        "request": request,
    }
    return sha256_text(canonical_json(attestation))


def parse_decisions(content: str, label_ids: list[str]) -> tuple[str, list[str], str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return "invalid_response", [], f"invalid_json:{exc.msg}"
    if not isinstance(payload, dict) or set(payload) != {"decisions"}:
        return "invalid_response", [], "top_level_schema_mismatch"
    decisions = payload["decisions"]
    if not isinstance(decisions, dict) or set(decisions) != set(label_ids):
        return "invalid_response", [], "decision_keys_mismatch"
    for label_id in label_ids:
        value = decisions[label_id]
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            return "invalid_response", [], f"nonbinary_decision:{label_id}"
    return "ok", [label_id for label_id in label_ids if decisions[label_id] == 1], ""


def execute_request(
    *,
    provider_name: str,
    provider: dict[str, Any],
    request: dict[str, Any],
    request_hash: str,
    label_ids: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    base = {
        "request_sha256": request_hash,
        "configured_model": str(provider["model"]),
        "requested_at_utc": utc_now(),
    }
    try:
        from openai import OpenAI

        key = os.environ.get(str(provider["key_env"]), "").strip()
        if not key:
            raise RuntimeError(f"missing API key environment variable {provider['key_env']}")
        client = OpenAI(
            api_key=key,
            base_url=str(provider["base_url"]),
            timeout=float(provider.get("request_timeout_seconds", 300)),
            max_retries=0,
        )
        completion = client.chat.completions.create(**request)
        message = completion.choices[0].message.content or ""
        status, predicted, format_error = parse_decisions(message, label_ids)
        usage = getattr(completion, "usage", None)
        return {
            **base,
            "status": status,
            "predicted_label_ids": predicted,
            "format_error": format_error,
            "raw_content": message,
            "actual_model": str(getattr(completion, "model", "")),
            "finish_reason": str(getattr(completion.choices[0], "finish_reason", "")),
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # external APIs expose provider-specific exception classes
        message = f"{type(exc).__name__}: {exc}"
        return {
            **base,
            "status": "api_failed",
            "failure_category": classify_failure(message),
            "error": message,
            "elapsed_seconds": time.perf_counter() - started,
        }
