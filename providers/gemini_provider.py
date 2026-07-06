import httpx

from providers.base import BaseProvider, ProviderError
from models import Message


class GeminiProvider(BaseProvider):

    name = "gemini"

    cost_per_1k_prompt = 0.00125
    cost_per_1k_completion = 0.005

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def chat(self, messages, model="gemini-2.5-flash", temperature=0.7, max_tokens=1024):

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )

        prompt = "\n".join(
            [f"{m.role}: {m.content}" for m in messages]
        )

        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=payload)

        except httpx.TimeoutException:
            raise ProviderError("Gemini timeout", retryable=True)

        except httpx.ConnectError:
            raise ProviderError("Gemini connection failed", retryable=True)

        if resp.status_code == 401:
            raise ProviderError("Gemini auth", retryable=False)

        if resp.status_code == 400:
            raise ProviderError(resp.text, retryable=False)

        if resp.status_code == 429:
            raise ProviderError("Gemini rate limit", retryable=True)

        if resp.status_code >= 500:
            raise ProviderError("Gemini server", retryable=True)

        data = resp.json()

        return {
            "content":
                data["candidates"][0]["content"]["parts"][0]["text"],
            "tokens_prompt":
                data["usageMetadata"]["promptTokenCount"],
            "tokens_completion":
                data["usageMetadata"]["candidatesTokenCount"],
            "model": model,
        }