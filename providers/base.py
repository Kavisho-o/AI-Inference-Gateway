"""
Defines the common interface every LLM provider adapter must implement.

The Router never talks to OpenAI, Anthropic, Gemini, etc. directly.
Instead, every provider subclasses BaseProvider and exposes the same
`chat()` method, allowing the gateway to swap providers without changing
routing logic.

Also defines ProviderError, which classifies provider failures as
retryable (timeouts, 429s, 5xx) or non-retryable (400, 401), enabling
the Router's retry and failover behaviour.
"""

from abc import ABC, abstractmethod
from models import Message

class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = True,status_code: int | None = None, provider: str | None = None,):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.provider = provider

class BaseProvider(ABC):
    name: str
    cost_per_1k_prompt: float = 0.0
    cost_per_1k_completion: float = 0.0

    @abstractmethod
    async def chat(self, messages: list[Message], model: str, temperature: float, max_tokens: int) -> dict:
        """
        Must return: {"content": str, "tokens_prompt": int, "tokens_completion": int, "model": str}
        Must raise ProviderError on failure, with retryable=True/False set correctly.
        Non-retryable: auth errors (401), bad request (400)
        Retryable: rate limit (429), timeout, 5xx
        """
        ...

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return ((prompt_tokens / 1000) * self.cost_per_1k_prompt + (completion_tokens / 1000) * self.cost_per_1k_completion)