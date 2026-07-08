"""
OpenAI provider adapter.

Converts the gateway's generic request format into the OpenAI Chat
Completions API, executes the request through the shared HTTP client,
normalizes the response into the gateway's standard schema, and maps
OpenAI-specific failures into ProviderError so the Router can retry or
fail over correctly.

Also exposes OpenAI pricing for cost estimation.
"""

import httpx
from providers.http_client import get_client
from providers.base import BaseProvider, ProviderError
from models import Message

class OpenAIProvider(BaseProvider):
    name = "openai"
    cost_per_1k_prompt = 0.0025   # gpt-4o-mini pricing, adjust as needed
    cost_per_1k_completion = 0.01

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def chat(self, messages: list[Message], model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1024) -> dict:
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        client = get_client()
        try:
            resp = await client.post(self.base_url, headers=headers, json=payload)
        except httpx.TimeoutException:
            raise ProviderError("OpenAI request timed out", retryable=True)
        except httpx.ConnectError:
            raise ProviderError("OpenAI connection failed", retryable=True)
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI request failed: {e}", retryable=True)

        if resp.status_code == 401:
            raise ProviderError("OpenAI auth failed", retryable=False)
        if resp.status_code == 400:
            raise ProviderError(f"OpenAI bad request: {resp.text}", retryable=False)
        if resp.status_code == 429:
            raise ProviderError("OpenAI rate limited", retryable=True)
        if resp.status_code >= 500:
            raise ProviderError(f"OpenAI server error: {resp.status_code}", retryable=True)
        if resp.status_code != 200:
            raise ProviderError(f"OpenAI unexpected status {resp.status_code}", retryable=True)

        data = resp.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "model": model,
            "tokens_prompt": data["usage"]["prompt_tokens"],
            "tokens_completion": data["usage"]["completion_tokens"],
        }