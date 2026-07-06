pipeline

client 
   |
FastAPI gateway 
   |
[rate limiter]
   | 
[cache check] 
   |
[router] 
   |
provider adapters 
   |
[OpenAI | Anthropic | Gemini | Ollama]

↓ on failure
[circuit breaker] 
   |
[retry w/ backoff] 
   |
[fallback provider]