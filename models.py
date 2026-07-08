"""
Shared Pydantic models used throughout the gateway.

Defines the standard request and response schema exchanged between
clients, the API layer, the Router, and provider adapters, ensuring all
providers operate on a common data format.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal

class Message(BaseModel):
    role: Literal["system","user","assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False

class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # FIX: silences the model_used warning
    content: str
    provider_used: str
    model_used: str
    latency_ms: float
    cached: bool = False
    tokens_prompt: int = 0
    tokens_completion: int = 0
    estimated_cost_usd: float = 0.0