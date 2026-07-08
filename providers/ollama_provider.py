"""
Ollama provider adapter.

Routes gateway requests to a locally running Ollama instance, converts
the response into the gateway's standard format, and wraps connection
or timeout failures as ProviderError. Since Ollama currently does not
return token usage, token counts are reported as zero.
"""

import httpx
from providers.base import BaseProvider, ProviderError

class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url):
        self.url = f"{base_url}/api/chat"

    async def chat(self, messages, model="llama3.2", temperature=0.7, max_tokens=1024):

        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(self.url, json=payload)

        except httpx.TimeoutException:
            raise ProviderError("Ollama timeout", retryable=True)

        except httpx.ConnectError:
            raise ProviderError("Ollama unavailable", retryable=True)

        if resp.status_code != 200:
            raise ProviderError(resp.text)

        data = resp.json()
        return {
            "content": data["message"]["content"],
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "model": model
        }