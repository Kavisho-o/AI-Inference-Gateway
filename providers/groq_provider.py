"""
Groq provider adapter.

Converts the gateway's generic request format into the Groq Chat
Completions API, executes the request through the shared HTTP client,
normalizes the response into the gateway's standard schema, and maps
Groq-specific failures into ProviderError so the Router can retry or
fail over correctly.

Also exposes Groq pricing for cost estimation.
"""

import httpx

from providers.base import BaseProvider, ProviderError
from providers.http_client import get_client
from models import Message


class GroqProvider(BaseProvider):
    name = "groq"

    # Update these if you use a different Groq model/pricing.
    cost_per_1k_prompt = 0.0
    cost_per_1k_completion = 0.0

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def chat(
        self,
        messages: list[Message],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        client = get_client()

        try:
            resp = await client.post(
                self.base_url,
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException:
            raise ProviderError("Groq request timed out", retryable=True)
        except httpx.ConnectError:
            raise ProviderError("Groq connection failed", retryable=True)
        except httpx.HTTPError as e:
            raise ProviderError(f"Groq request failed: {e}", retryable=True)

        if resp.status_code == 401:
            raise ProviderError("Groq auth failed", retryable=False)

        if resp.status_code == 400:
            raise ProviderError(f"Groq bad request: {resp.text}", retryable=False)

        if resp.status_code == 429:
            raise ProviderError("Groq rate limited", retryable=True)

        if resp.status_code >= 500:
            raise ProviderError(
                f"Groq server error: {resp.status_code}",
                retryable=True,
            )

        if resp.status_code != 200:
            raise ProviderError(
                f"Groq unexpected status {resp.status_code}",
                retryable=True,
            )

        data = resp.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", model),
            "tokens_prompt": data["usage"]["prompt_tokens"],
            "tokens_completion": data["usage"]["completion_tokens"],
        }