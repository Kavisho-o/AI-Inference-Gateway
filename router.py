"""
Core orchestration layer of the inference gateway.

Coordinates request execution by checking the cache, selecting providers
using the configured fallback chain, enforcing circuit breakers,
performing retries with exponential backoff, estimating request cost,
recording usage statistics, and caching successful responses before
returning a unified result.
"""

import asyncio
import time
from providers.base import ProviderError
from circuit_breaker import CircuitBreaker
from cache import ResponseCache
from cost_tracker import CostTracker
from config import get_settings

settings = get_settings()

class Router:
    def __init__(self, providers: dict, cache: ResponseCache, cost_tracker: CostTracker):
        self.providers = providers  # {"openai": OpenAIProvider(...), ...}
        self.breakers = {name: CircuitBreaker(
            settings.circuit_breaker_failure_threshold,
            settings.circuit_breaker_recovery_seconds
        ) for name in providers}
        self.cache = cache
        self.cost_tracker = cost_tracker

    async def route(self, request, explicit_provider: str | None):
        cache_key = self.cache.make_key(request)
        cached = await self.cache.get(cache_key)
        if cached:
            cached["cached"] = True
            return cached

        chain = [explicit_provider] if explicit_provider else settings.fallback_chain
        last_error = None

        for provider_name in chain:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            breaker = self.breakers[provider_name]
            if not breaker.allow_request():
                continue  # circuit open, skip to next in chain

            result = await self._call_with_retry(provider, request)
            if result is not None:
                breaker.record_success()
                result["provider_used"] = provider_name
                result["cost"] = provider.estimate_cost(
                    result["tokens_prompt"], result["tokens_completion"]
                )
                self.cost_tracker.record(provider_name, result["cost"])
                await self.cache.set(cache_key, result)
                return result
            breaker.record_failure()
            last_error = f"{provider_name} exhausted retries"

        raise RuntimeError(f"All providers in chain failed. Last error: {last_error}")

    async def _call_with_retry(self, provider, request):
        kwargs = {
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.model:  # FIX: only override the provider's own default when the client actually asked for one
            kwargs["model"] = request.model

        for attempt in range(settings.max_retries):
            try:
                start = time.monotonic()
                result = await provider.chat(**kwargs)
                result["latency_ms"] = (time.monotonic() - start) * 1000
                return result
            except ProviderError as e:
                if not e.retryable:
                    return None
                if attempt < settings.max_retries - 1:
                    await asyncio.sleep(settings.base_backoff_seconds * (2 ** attempt))
            except Exception as e: 
                print(f"[router] Unexpected error from {provider.name}: {e}")
                if attempt < settings.max_retries - 1:
                    await asyncio.sleep(settings.base_backoff_seconds * (2 ** attempt))
        return None