# inference-gateway

A single API endpoint that routes chat completions across OpenAI, Anthropic, Gemini, and a local Ollama model — with automatic failover when a provider is down, rate limited, or misconfigured, plus response caching, per-request cost tracking, and circuit breakers so a struggling provider gets skipped instead of retried into the ground.

Live at: [JUMP HERE](https://ai-inference-gateway.onrender.com/docs)

## What it does

1. Client sends one request to `/v1/chat/completions` — same shape regardless of which provider ends up answering
2. Request hits a response cache first — identical requests within the TTL window return instantly with zero provider calls
3. A token-bucket rate limiter checks the caller's API key before anything else runs
4. The router walks a fallback chain (`openai → anthropic → gemini → ollama` by default, or a single explicit provider if specified)
5. Each provider call goes through a circuit breaker — a provider that's failed repeatedly gets skipped entirely instead of retried, until a recovery window passes
6. Failed calls retry with exponential backoff up to a configured limit, but only for retryable failures (timeouts, 429s, 5xx) — auth failures and bad requests fail fast and move to the next provider immediately
7. On success: the response is cached, cost is estimated from actual token usage and logged per-provider, and the client gets back a consistent schema no matter which provider answered

## Why these design choices

**Fallback chain over a single hardcoded provider.** Any single LLM provider can go down, rate-limit you, or have a bad day. Routing through a chain means a client-facing failure only happens if every provider in the chain fails — not if the first one does.

**Circuit breaker over naive retries.** Retrying a genuinely down provider on every request wastes time and money on calls that are going to fail anyway. The breaker tracks failure counts per provider and opens after a threshold, skipping straight past that provider until a recovery window passes — cheaper and faster than discovering the same outage on every single request.

**Distinguishing retryable from non-retryable failures.** A 401 (bad key) or 400 (malformed request) will fail identically on every retry — retrying it just adds latency. A 429 or timeout might succeed a second later. Each provider tags its own errors with `retryable=True/False` so the router only spends retry budget where it can actually help.

**Response caching keyed on the full request.** Identical questions with identical parameters (model, temperature, max_tokens) get a hash-based cache hit, skipping the provider call entirely. This is the single biggest cost lever in a gateway like this — repeated identical prompts are common in real usage (health checks, dev testing, common questions).

**Per-provider cost tracking.** Different providers have wildly different pricing, and picking the "best" model without seeing cost per request is guessing. Tracking token usage and estimated cost per provider, per request, makes the tradeoff visible instead of assumed.

## Stack

`FastAPI` · `httpx` (async) · `pydantic` / `pydantic-settings` · `OpenAI` · `Anthropic` · `Gemini` · `Ollama` (local) · `Docker`

## Architecture

```text
                     CLIENT

                        │
                        ▼

              FastAPI Inference Gateway

                        │
                        ▼

                Authentication

                        │
                        ▼

                 Rate Limiter
             (Token Bucket Algorithm)

                        │
                        ▼

                 Request Cache

                        │
                 Cache Hit? ─────► Return immediately

                        │
                 Cache Miss

                        ▼

                  Router

                        │

          ┌─────────────┼──────────────┐

          ▼             ▼              ▼

     OpenAI       Anthropic      Gemini

          │             │             │

          ▼             ▼             ▼

     Circuit Breaker per Provider

          │

          ▼

   Retry + Exponential Backoff

          │

          ▼

      Successful Response

          │

          ▼

 Cost Tracker + Cache Response

          │

          ▼

       Return to Client
```

## Bugs worth knowing about

**The app couldn't import at all.** `config.py` had `fallback_chain = [...]` with no type annotation inside a Pydantic v2 `BaseSettings` class. Pydantic v2 requires every class attribute in a settings model to be annotated, or it raises `PydanticUserError` at class-definition time — meaning the app failed before `main.py` even finished importing, not at runtime. Fixed by annotating it as `fallback_chain: list[str]`.

**Two providers never actually implemented their interface.** `BaseProvider` declares `chat()` as the abstract method every provider must implement. The OpenAI and Anthropic adapters had both been written with a method named `generate()` instead — a naming mismatch that meant Python's ABC machinery refused to instantiate either class (`TypeError: Can't instantiate abstract class ... with abstract method chat`). This broke at server startup, not per-request, so nothing served at all until it was caught.

**A shared HTTP client that was never started.** `providers/http_client.py` holds a shared `httpx.AsyncClient`, initialized via a `startup()` function meant to run once at app boot. That function was defined but never called anywhere in the app's lifespan. Every OpenAI and Anthropic call raised `RuntimeError: HTTP client not initialized` — and because that's a plain `RuntimeError` rather than the app's own `ProviderError` type, it wasn't caught by the retry logic, so it broke the fallback chain instead of just failing over to the next provider.

**An undefined variable that only fails on a real response.** The Anthropic provider referenced a variable called `text` when building its return value — except `text` was never assigned anywhere in the function. Anthropic's actual content lives at `response["content"][0]["text"]`; the extraction step had been skipped entirely. This wouldn't surface in a quick code read — it only throws once a real API call comes back successfully.

**One error type silently broke the whole point of a fallback chain.** The retry logic only caught the app's own `ProviderError` exceptions. Any other unexpected error — a malformed response, a parsing bug like the one above — propagated straight past the retry loop and the fallback chain, killing the entire request instead of moving on to the next provider. Widened the catch to treat any unexpected exception as a failed attempt, so one provider's bug degrades gracefully instead of taking the whole request down with it.

**Router was overriding every provider's sensible default model.** When a client didn't specify a model, the router passed the literal string `"default"` to whichever provider got called — not a real model name for any of the four providers, so every default-model request would have failed with a bad-request error. Fixed by only passing a `model` argument when the client actually specified one, letting each provider fall back to its own real default.

## Running locally

```bash
pip install -r requirements.txt

# .env file:
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...
# GEMINI_API_KEY=...
# OLLAMA_BASE_URL=http://localhost:11434

uvicorn main:app --reload
```

Requires [Ollama](https://ollama.com) running locally for the `ollama` provider (`ollama pull llama3.2`), and free-tier API keys for any paid providers you want to test.

## Testing

Full endpoint testing done via Swagger UI (`/docs`) and PowerShell, covering: successful completions per provider, cache-hit verification, full fallback chain traversal on missing/bad credentials, rate limiter enforcement (429 after burst), and circuit breaker tripping + recovery on a deliberately broken provider.
