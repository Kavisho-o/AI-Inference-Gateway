import asyncio
import httpx
import time

URL = "http://localhost:8000/v1/chat/completions"

payload = {
    "messages": [{"role": "user", "content": "Explain recursion in one sentence."}]
}

async def hit(i):

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        r = await client.post(URL, json=payload)

    latency = (time.perf_counter() - start) * 1000
    body = r.json()

    print(
        f"{i:02d} | "
        f"{body['provider_used']} | "
        f"{latency:.2f} ms | "
        f"cached={body['cached']}"
    )

    return latency


async def main():

    tasks = [hit(i) for i in range(10)]
    latencies = await asyncio.gather(*tasks)
    print()
    print("Average:", sum(latencies) / len(latencies))
    print("Fastest:", min(latencies))
    print("Slowest:", max(latencies))


asyncio.run(main())