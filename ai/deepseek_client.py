"""DeepSeek API client. OpenAI-compatible format.

Uses deepseek-v4-flash (non-thinking) — latest stable, fast and cheap,
suitable for cover letter generation and vacancy relevance scoring.
"""
import asyncio
import logging
import time

import httpx

from config import DEEPSEEK_API_KEY, GLM_PROXY

logger = logging.getLogger(__name__)

API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
PRO_MODEL = "deepseek-v4-pro"

# Soft cooldown — paid tier is generous, but keep a small floor to
# avoid hammering during burst loops.
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 1.0  # seconds


async def deepseek_chat(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 2000,
    model: str | None = None,
    thinking: bool = False,
    response_format: dict | None = None,
) -> str:
    """Send chat completion request to DeepSeek API. Returns content string.

    Args:
        model: defaults to deepseek-v4-flash. Pass PRO_MODEL for higher quality
            (recommended together with thinking=True for cover letters).
        thinking: enable reasoning mode. Slower and pricier, but writes deeper.
    """
    global _last_request_time

    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is empty")

    now = time.time()
    wait_needed = MIN_REQUEST_INTERVAL - (now - _last_request_time)
    if wait_needed > 0:
        await asyncio.sleep(wait_needed)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }
    if response_format:
        payload["response_format"] = response_format

    transport = None
    if GLM_PROXY:
        transport = httpx.AsyncHTTPTransport(proxy=GLM_PROXY)

    async with httpx.AsyncClient(transport=transport, timeout=120) as client:
        for attempt in range(3):
            try:
                _last_request_time = time.time()
                resp = await client.post(API_URL, json=payload, headers=headers)
            except httpx.TimeoutException:
                logger.warning("DeepSeek timeout (attempt %d/3)", attempt + 1)
                await asyncio.sleep(10)
                continue
            except httpx.HTTPError as e:
                logger.warning("DeepSeek HTTP error (attempt %d/3): %s", attempt + 1, e)
                await asyncio.sleep(10)
                continue

            if resp.status_code == 429:
                wait = 20 * (attempt + 1)  # 20s, 40s, 60s
                logger.warning(
                    "DeepSeek 429, waiting %ds (attempt %d/3)", wait, attempt + 1
                )
                await asyncio.sleep(wait)
                continue

            if resp.status_code >= 500:
                wait = 10 * (attempt + 1)
                logger.warning(
                    "DeepSeek %d server error, waiting %ds (attempt %d/3)",
                    resp.status_code, wait, attempt + 1,
                )
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "choices" not in data:
                logger.error("DeepSeek unexpected response: %s", str(data)[:300])
                raise RuntimeError(f"DeepSeek unexpected: {str(data)[:200]}")

            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content:
                # In thinking mode V4 may put the whole answer into
                # reasoning_content. Fall back to it instead of failing.
                content = (msg.get("reasoning_content") or "").strip()
            if not content:
                logger.warning("DeepSeek returned empty content and reasoning")
                raise RuntimeError("DeepSeek returned empty response")
            return content

        raise RuntimeError("DeepSeek API failed after 3 retries")
