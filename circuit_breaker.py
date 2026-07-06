import time
import threading
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # failing, reject immediately
    HALF_OPEN = "half_open" # testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_seconds: int):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at: float | None = None
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.monotonic() - self.opened_at >= self.recovery_seconds:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            if self.state == CircuitState.HALF_OPEN:
                return True
        return False

    def record_success(self):
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def record_failure(self):
        with self.lock:
            self.failure_count += 1
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()