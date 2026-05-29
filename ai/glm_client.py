"""GLM API client via z.ai proxy endpoint with SOCKS5 proxy support."""
import asyncio
import logging
import time
import jwt
import httpx
from config import GLM_API_KEY, GLM_PROXY

logger = logging.getLogger(__name__)

API_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
MODEL = "glm-5"


def _make_token() -> str:
    """Generate JWT token for Zhipu AI API."""
    parts = GLM_API_KEY.split(".")
    if len(parts) != 2:
        raise ValueError("GLM_API_KEY must be in format 'kid.secret'")
    kid, secret = parts
    payload = {
        "api_key": kid,
        "exp": int(time.time()) + 600,
        "timestamp": int(time.time()),
    }
    return jwt.encode(
        payload, secret, algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


def _extract_text(data: dict) -> str:
    """Extract text from GLM response. GLM-5 is a reasoning model:
    it puts thinking in reasoning_content and final answer in content.
    Sometimes content is empty and answer is only in reasoning."""
    msg = data["choices"][0]["message"]
    content = msg.get("content", "").strip()
    reasoning = msg.get("reasoning_content", "").strip()

    # prefer content (final answer)
    if content:
        return content

    # GLM-5 sometimes puts entire answer in reasoning_content
    if reasoning:
        logger.info("GLM: content empty, using reasoning_content")
        return reasoning

    return ""


async def glm_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 4000) -> str:
    """Send chat completion request to GLM API. Returns content string."""
    token = _make_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    transport = None
    if GLM_PROXY:
        transport = httpx.AsyncHTTPTransport(proxy=GLM_PROXY)

    async with httpx.AsyncClient(transport=transport, timeout=300) as client:
        for attempt in range(3):
            try:
                resp = await client.post(API_URL, json=payload, headers=headers)
            except httpx.TimeoutException:
                logger.warning("GLM timeout (attempt %d/3)", attempt + 1)
                await asyncio.sleep(30)
                continue

            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                logger.warning("GLM 429 rate limit, waiting %ds (attempt %d/3)", wait, attempt + 1)
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "choices" not in data:
                logger.error("GLM unexpected response: %s", str(data)[:300])
                raise RuntimeError(f"GLM unexpected response: {str(data)[:200]}")

            result = _extract_text(data)
            if not result:
                logger.warning("GLM returned empty content and reasoning")
                raise RuntimeError("GLM returned empty response")
            return result

        raise RuntimeError("GLM API failed after 3 retries")
