"""
FastAPI application entry point.

Initializes shared infrastructure during startup (providers, router,
cache, rate limiter, cost tracker, and HTTP client), exposes the public
chat completion endpoint, enforces rate limiting, and returns a
provider-agnostic response regardless of which LLM generated it.

Also exposes health and usage statistics endpoints.
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from contextlib import asynccontextmanager
from models import ChatRequest, ChatResponse
from providers.anthropic_provider import AnthropicProvider
from providers.gemini_provider import GeminiProvider
from providers.ollama_provider import OllamaProvider
from providers.groq_provider import GroqProvider
from rate_limiter import RateLimiter
from cache import ResponseCache
from cost_tracker import CostTracker
from router import Router
from providers.openai_provider import OpenAIProvider
from providers import http_client  # FIX: needed to actually start/stop the shared client
from config import get_settings

settings = get_settings()
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await http_client.startup()  # FIX: was never called — get_client() raised RuntimeError on every OpenAI/Anthropic request

    providers = {
        "openai": OpenAIProvider(settings.openai_api_key),
        "anthropic": AnthropicProvider(settings.anthropic_api_key),
        "gemini": GeminiProvider(settings.gemini_api_key),
        "ollama": OllamaProvider(settings.ollama_base_url),
        "groq": GroqProvider(settings.groq_api_key)
    }
    cache = ResponseCache(settings.cache_ttl_seconds)
    cost_tracker = CostTracker()
    state["rate_limiter"] = RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
    state["router"] = Router(providers, cache, cost_tracker)
    state["cost_tracker"] = cost_tracker
    yield
    await http_client.shutdown()  # FIX: clean shutdown of the shared client
    state.clear()

app = FastAPI(title="AI Inference Gateway", lifespan=lifespan)

def get_api_key(x_api_key: str = Header(default="anonymous")):
    return x_api_key

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest, api_key: str = Depends(get_api_key)):
    if not state["rate_limiter"].check(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    try:
        result = await state["router"].route(request, request.provider)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return ChatResponse(
        content=result["content"],
        provider_used=result["provider_used"],
        model_used=result["model"],
        latency_ms=result.get("latency_ms", 0.0),
        cached=result.get("cached", False),
        tokens_prompt=result["tokens_prompt"],
        tokens_completion=result["tokens_completion"],
        estimated_cost_usd=result.get("cost", 0.0),
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/stats")
async def stats():
    return state["cost_tracker"].summary()