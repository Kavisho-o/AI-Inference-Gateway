"""
Creates and manages a shared httpx.AsyncClient used by provider adapters.

Instead of creating a new HTTP connection for every request, the gateway
initializes one reusable client during FastAPI startup and closes it
during shutdown, enabling connection pooling, lower latency, and reduced
resource usage.
"""

import httpx

_client: httpx.AsyncClient | None = None

async def startup():
    global _client

    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    )


async def shutdown():
    global _client

    if _client:
        await _client.aclose()
        _client = None


def get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("HTTP client not initialized")

    return _client