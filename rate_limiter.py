"""
Implements API rate limiting using the Token Bucket algorithm.

Each API key receives an independent token bucket, allowing controlled
burst traffic while enforcing a sustained request rate to protect the
gateway from abuse.
"""

import time
import threading
from collections import defaultdict

class TokenBucket:
    def __init__(self, rate_per_minute: int, burst: int):
        self.rate = rate_per_minute / 60.0   # tokens per second
        self.burst = burst
        self.tokens = float(burst)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

class RateLimiter:
    def __init__(self, rate_per_minute: int, burst: int):
        self.rate_per_minute = rate_per_minute
        self.burst = burst
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.rate_per_minute, self.burst)
        )
        self._buckets_lock = threading.Lock()

    def check(self, api_key: str) -> bool:
        with self._buckets_lock:
            bucket = self.buckets[api_key]
        return bucket.allow()