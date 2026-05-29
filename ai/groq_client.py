"""Groq API client. OpenAI-compatible format, no proxy needed."""
import asyncio
import logging
import httpx
from config import GROQ_API_KEY

logger = logging.getLogger(__name__)

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

# Global cooldown to avoid 429
_last_request_time = 0
MIN_REQUEST_INTERVAL = 10  # seconds between requests


async def groq_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Send chat completion request to Groq API. Returns content string."""
    import time
    global _last_request_time

    # enforce minimum interval between requests
    now = time.time()
    wait_needed = MIN_REQUEST_INTERVAL - (now - _last_request_time)
    if wait_needed > 0:
        await asyncio.sleep(wait_needed)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(3):
            try:
                _last_request_time = time.time()
                resp = await client.post(API_URL, json=payload, headers=headers)
            except httpx.TimeoutException:
                logger.warning("Groq timeout (attempt %d/3)", attempt + 1)
                await asyncio.sleep(15)
                continue

            if resp.status_code == 429:
                wait = 180 * (attempt + 1)  # 3min, 6min, 9min
                logger.warning("Groq 429, waiting %ds (attempt %d/3)", wait, attempt + 1)
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "choices" not in data:
                logger.error("Groq unexpected response: %s", str(data)[:300])
                raise RuntimeError(f"Groq unexpected: {str(data)[:200]}")

            content = data["choices"][0]["message"].get("content", "").strip()
            if not content:
                logger.warning("Groq returned empty content")
                raise RuntimeError("Groq returned empty response")
            return content

        raise RuntimeError("Groq API failed after 3 retries")
