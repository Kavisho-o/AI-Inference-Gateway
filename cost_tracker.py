import threading
from collections import defaultdict

class CostTracker:
    def __init__(self):
        self.costs: dict[str, float] = defaultdict(float)
        self.request_counts: dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def record(self, provider: str, cost: float):
        with self.lock:
            self.costs[provider] += cost
            self.request_counts[provider] += 1

    def summary(self) -> dict:
        with self.lock:
            return {
                "total_cost_usd": round(sum(self.costs.values()), 6),
                "by_provider": {k: round(v, 6) for k, v in self.costs.items()},
                "request_counts": dict(self.request_counts),
            }