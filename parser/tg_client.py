"""Telegram channel reader via the public web mirror (t.me/s/<channel>).

Why not Telethon/MTProto: that path needs api_id/api_hash from
my.telegram.org, which is region-blocked for some accounts (the infamous
"ERROR" on app creation). Public channels expose a credential-free web
preview at https://t.me/s/<username> — a plain HTML page with the latest
posts. We fetch and parse that. No login, no session file, no api keys.

Trade-off: the web mirror only exposes RECENT posts (~last 15-20 per
channel), not deep history. That's exactly what a periodic monitor needs
(it polls every N minutes and only cares about new posts), so it's a fit.
Channels that disabled their web preview return nothing — they're logged
and skipped.

This module owns:
  * loading the channel list from data/tg_channels.yaml
  * fetching + parsing recent posts into TgPost objects

It deliberately does NOT decide if a post is a vacancy (ai/tg_extractor.py),
persist anything (bot/tg_loop.py), or score relevance (ai/analyzer.py).

The public interface (load_channels / fetch_recent_posts / ChannelSpec /
TgPost) is unchanged from the old Telethon version, so bot/tg_loop.py and
scripts/tg_pull_test.py work without modification.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import aiohttp
import yaml
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

# A real browser UA — t.me sometimes serves a stripped page to unknown
# clients. Pinning a desktop Chrome UA keeps the message markup stable.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru,en;q=0.9",
}
_CHANNEL_PAUSE = 1.0  # seconds between channel fetches (be polite)
_REQUEST_TIMEOUT = 30  # seconds per channel page


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ChannelSpec:
    """One channel entry from data/tg_channels.yaml."""

    username: str
    role_filter: list[str] = field(default_factory=list)
    enabled: bool = True
    note: str = ""

    def matches_prefilter(self, text: str) -> bool:
        """Quick keyword pre-filter BEFORE calling the LLM.

        If `role_filter` is empty we accept everything (LLM will decide).
        Otherwise the post must mention at least one of the listed
        keywords (case-insensitive substring match).
        """
        if not self.role_filter:
            return True
        haystack = text.lower()
        return any(kw.lower() in haystack for kw in self.role_filter)


@dataclass
class TgPost:
    """Normalised representation of one Telegram message."""

    id: str            # "tg:<channel>:<message_id>"
    source: str        # always "telegram"
    channel: str       # channel username (without @)
    message_id: int
    text: str
    url: str           # https://t.me/<channel>/<message_id>
    posted_at: datetime
    views: int | None = None

    def as_vacancy_dict(self) -> dict:
        """Shape it like the dicts that analyzer.py / cover_letter.py expect.

        Title and company are unknown at this stage — extraction happens in
        a separate LLM step (ai/tg_extractor.py). For now we put the channel
        name as `company` so the analyzer prompt has something to anchor on,
        and leave `title` empty.
        """
        return {
            "id": self.id,
            "source": self.source,
            "channel": self.channel,
            "title": "",
            "company": f"@{self.channel}",
            "description": self.text,
            "url": self.url,
            "posted_at": self.posted_at.isoformat(),
            "views": self.views,
        }


# ---------------------------------------------------------------------------
# Channel config loading
# ---------------------------------------------------------------------------
def load_channels(path: Path | None = None) -> list[ChannelSpec]:
    """Load the channel list from YAML config.

    Disabled channels are filtered out automatically.
    """
    path = path or config.TG_CHANNELS_CONFIG
    if not path.exists():
        logger.error("Channel config not found: %s", path)
        return []

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    channels_raw = raw.get("channels", [])
    out: list[ChannelSpec] = []
    for entry in channels_raw:
        if not entry.get("username"):
            continue
        spec = ChannelSpec(
            username=entry["username"].lstrip("@"),
            role_filter=entry.get("role_filter", []) or [],
            enabled=entry.get("enabled", True),
            note=entry.get("note", ""),
        )
        if spec.enabled:
            out.append(spec)
    logger.info("Loaded %d enabled channel(s) from %s", len(out), path.name)
    return out


# ---------------------------------------------------------------------------
# Fetching via the t.me/s/ web mirror
# ---------------------------------------------------------------------------
async def fetch_recent_posts(
    channels: Iterable[ChannelSpec],
    *,
    since: datetime | None = None,
    limit_per_channel: int = 30,
) -> list[TgPost]:
    """Pull recent posts from each channel's web mirror.

    Parameters
    ----------
    channels : iterable of ChannelSpec
        Channels to scan (disabled ones already filtered by load_channels).
    since : datetime, optional
        Only return messages newer than this. Defaults to 48 hours ago.
        Must be timezone-aware (UTC); a naive value is assumed to be UTC.
    limit_per_channel : int
        Keep at most this many (most-recent) posts per channel.

    Returns
    -------
    list[TgPost]
        Posts that passed the per-channel keyword pre-filter. NOT yet
        analysed by LLM — that happens downstream.
    """
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(hours=48)
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    proxy = config.TG_HTTP_PROXY or None
    out: list[TgPost] = []
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout, headers=_HEADERS) as session:
        for spec in channels:
            posts = await _fetch_one_channel(
                session, spec, since=since, limit=limit_per_channel, proxy=proxy
            )
            out.extend(posts)
            await asyncio.sleep(_CHANNEL_PAUSE)
    return out


async def _fetch_one_channel(
    session: aiohttp.ClientSession,
    spec: ChannelSpec,
    *,
    since: datetime,
    limit: int,
    proxy: str | None,
) -> list[TgPost]:
    """Fetch + parse one channel's web mirror with per-channel error isolation."""
    url = f"https://t.me/s/{spec.username}"
    try:
        async with session.get(url, proxy=proxy) as resp:
            if resp.status != 200:
                logger.warning(
                    "t.me/s/%s -> HTTP %s (channel dead or web-preview off)",
                    spec.username, resp.status,
                )
                return []
            html = await resp.text()
    except Exception as e:  # noqa: BLE001 — last-resort per-channel isolation
        logger.warning("Failed to fetch @%s: %s", spec.username, e)
        return []

    parsed = _parse_messages(html, spec.username)
    if not parsed:
        logger.info(
            "Channel @%s: 0 posts parsed (empty page or web-preview disabled)",
            spec.username,
        )
        return []

    # Filter by recency + keyword pre-filter, then keep the most recent `limit`.
    kept: list[TgPost] = []
    for p in parsed:  # parsed is in document order: oldest -> newest
        if p.posted_at < since:
            continue
        if not spec.matches_prefilter(p.text):
            continue
        kept.append(p)
    if limit and len(kept) > limit:
        kept = kept[-limit:]

    logger.info("Channel @%s: %d post(s) after filter", spec.username, len(kept))
    return kept


def _parse_messages(html: str, channel: str) -> list[TgPost]:
    """Parse the t.me/s/ HTML into TgPost objects (document order)."""
    soup = BeautifulSoup(html, "html.parser")
    posts: list[TgPost] = []

    for node in soup.select("div.tgme_widget_message"):
        data_post = node.get("data-post")  # "<channel>/<message_id>"
        if not data_post or "/" not in data_post:
            continue
        try:
            msg_id = int(data_post.rsplit("/", 1)[-1])
        except ValueError:
            continue

        text_node = node.select_one(".tgme_widget_message_text")
        if text_node is None:
            continue  # media-only post, no caption — nothing to analyse
        # Preserve line breaks: Telegram uses <br> inside the text block.
        for br in text_node.find_all("br"):
            br.replace_with("\n")
        text = text_node.get_text().strip()
        if not text:
            continue

        time_node = node.select_one("time[datetime]")
        posted_at = _parse_dt(time_node.get("datetime")) if time_node else None
        if posted_at is None:
            # No timestamp in markup — treat as "now" so it isn't dropped by
            # the recency filter; tg_seen dedup prevents re-notifying.
            posted_at = datetime.now(timezone.utc)

        posts.append(
            TgPost(
                id=f"tg:{channel}:{msg_id}",
                source="telegram",
                channel=channel,
                message_id=msg_id,
                text=text,
                url=f"https://t.me/{channel}/{msg_id}",
                posted_at=posted_at,
                views=None,
            )
        )

    return posts


def _parse_dt(value: str | None) -> datetime | None:
    """Parse the ISO datetime from a <time datetime="..."> attribute."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# CLI entry point — `python -m parser.tg_client test`
# ---------------------------------------------------------------------------
def _main() -> None:
    import sys

    # Windows consoles default to cp1251 and mojibake Cyrillic on print().
    # Force UTF-8 so the manual test output is readable.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) >= 2 and sys.argv[1] not in ("test", "pull"):
        print("Usage: python -m parser.tg_client [test]")
        sys.exit(2)

    channels = load_channels()
    if not channels:
        print("No enabled channels in data/tg_channels.yaml")
        return

    posts = asyncio.run(fetch_recent_posts(channels, limit_per_channel=10))
    print(f"\nFetched {len(posts)} post(s) from {len(channels)} channel(s).")
    for p in posts[:10]:
        preview = p.text[:120].replace("\n", " ")
        print(f"  [{p.channel}] {p.posted_at.date()} {p.url}\n    {preview}")


if __name__ == "__main__":
    _main()
