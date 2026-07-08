"""
Simple concurrent load test for the AI Inference Gateway.

Sends multiple requests to the /v1/chat/completions endpoint,
measures latency, reports which provider served each request,
and prints overall performance statistics.

Unlike a stress-testing tool, this is intended for quick local
performance verification while developing the gateway.
"""

import asyncio
import time

import httpx

URL = "http://localhost:8000/v1/chat/completions"

payload = {
    "messages": [
        {
            "role": "user",
            "content": "Explain recursion in one sentence."
        }
    ],
    # Use a working provider while benchmarking.
    # Change to "gemini" if you want to benchmark Gemini instead.
    "provider": "gemini"
}


async def hit(i: int):
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(URL, json=payload)

        latency = (time.perf_counter() - start) * 1000
        body = response.json()

        if response.status_code != 200:
            print(
                f"{i:02d} | FAILED | "
                f"{response.status_code} | "
                f"{body}"
            )
            return None

        print(
            f"{i:02d} | "
            f"{body['provider_used']:<10} | "
            f"{latency:8.2f} ms | "
            f"cached={body['cached']}"
        )

        return latency

    except Exception as e:
        latency = (time.perf_counter() - start) * 1000

        print(
            f"{i:02d} | "
            f"FAILED     | "
            f"{latency:8.2f} ms | "
            f"{type(e).__name__}: {e}"
        )

        return None


async def main():
    tasks = [hit(i) for i in range(5)]

    latencies = await asyncio.gather(*tasks)
    successful = [x for x in latencies if x is not None]
    print("\n" + "-" * 45)

    if successful:
        print(f"Successful Requests : {len(successful)}/{len(tasks)}")
        print(f"Average Latency     : {sum(successful)/len(successful):.2f} ms")
        print(f"Fastest Request     : {min(successful):.2f} ms")
        print(f"Slowest Request     : {max(successful):.2f} ms")
    else:
        print("No successful requests.")


if __name__ == "__main__":
    asyncio.run(main())