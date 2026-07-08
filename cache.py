"""
Simple in-memory response cache with TTL expiration.

Requests are hashed using their messages and generation parameters,
allowing identical requests to reuse previous responses instead of
calling an LLM again, reducing latency and API cost.
"""

import hashlib
import json
import time

class ResponseCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self.store: dict[str, tuple[float, dict]] = {}

    def make_key(self, request) -> str:
        payload = {
            "messages": [m.model_dump() for m in request.messages],
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, key: str) -> dict | None:
        entry = self.store.get(key)
        if not entry:
            return None
        timestamp, value = entry
        if time.time() - timestamp > self.ttl:
            del self.store[key]
            return None
        return dict(value)

    async def set(self, key: str, value: dict):
        self.store[key] = (time.time(), value)