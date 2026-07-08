"""
Anthropic provider adapter.

Transforms generic gateway requests into Anthropic's Messages API
format (including separate system prompts), performs the API request
using the shared HTTP client, normalizes the response, and converts
provider-specific failures into ProviderError for Router retry/fallback.

Also stores Anthropic pricing used by the CostTracker.
"""

import httpx

from providers.base import BaseProvider, ProviderError
from models import Message
from providers.http_client import get_client

class AnthropicProvider(BaseProvider):

    name = "anthropic"
    cost_per_1k_prompt = 0.003
    cost_per_1k_completion = 0.015

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.anthropic.com/v1/messages"

    async def chat(self, messages: list[Message], model="claude-3-5-haiku-latest", temperature=0.7, max_tokens=1024):
        system_parts = [m.content for m in messages if m.role == "system"]
        chat_messages = [m for m in messages if m.role != "system"]

        payload = {
            "model": model,
            "messages": [m.model_dump() for m in chat_messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        try:
            client = get_client()
            resp = await client.post(self.url, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise ProviderError("Anthropic timeout", retryable=True)
        except httpx.ConnectError:
            raise ProviderError("Anthropic connection failed", retryable=True)
        except httpx.HTTPError as e:
            raise ProviderError(f"Anthropic request failed: {e}", retryable=True)

        if resp.status_code == 401:
            raise ProviderError("Anthropic auth failed", retryable=False)
        if resp.status_code == 400:
            raise ProviderError(resp.text, retryable=False)
        if resp.status_code == 429:
            raise ProviderError("Anthropic rate limited", retryable=True)
        if resp.status_code >= 500:
            raise ProviderError("Anthropic server error", retryable=True)

        data = resp.json()
        content = data["content"][0]["text"]  # FIX: 'text' was undefined before — this is where it actually lives

        return {
            "content": content,
            "model": model,
            "tokens_prompt": data["usage"]["input_tokens"],
            "tokens_completion": data["usage"]["output_tokens"],
        }