"""One-off manual test of the Telegram channel parser.

Pulls recent messages from every enabled channel in data/tg_channels.yaml
(applying the per-channel keyword pre-filter), dumps them to a JSON file
and a human-readable Markdown report, and prints a summary.

This is intentionally NOT integrated with the bot or the analyzer yet —
the point is to look at what the raw pre-filtered stream looks like
BEFORE we add LLM analysis, so we can tune `role_filter` per channel
based on real data.

Run:
    python -m scripts.tg_pull_test
or:
    python scripts/tg_pull_test.py

Prerequisites:
    1. `pip install -r requirements.txt` (aiohttp + beautifulsoup4).
    No api keys, no login — channels are read via the public web mirror
    (https://t.me/s/<channel>). Just make sure the usernames in
    data/tg_channels.yaml are correct and the channels have web preview on.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running both as a module and as a plain script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parser.tg_client import fetch_recent_posts, load_channels  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tg_pull_test")


HOURS_BACK = 48
LIMIT_PER_CHANNEL = 30

OUT_JSON = ROOT / "data" / "tg_pull_test.json"
OUT_MARKDOWN = ROOT / "data" / "tg_pull_test.md"


async def main() -> None:
    channels = load_channels()
    if not channels:
        logger.error(
            "No enabled channels in %s — nothing to pull.",
            "data/tg_channels.yaml",
        )
        return

    logger.info(
        "Pulling last %d hours from %d channel(s), max %d posts each",
        HOURS_BACK,
        len(channels),
        LIMIT_PER_CHANNEL,
    )

    since = datetime.now(timezone.utc) - timedelta(hours=HOURS_BACK)
    posts = await fetch_recent_posts(
        channels,
        since=since,
        limit_per_channel=LIMIT_PER_CHANNEL,
    )

    if not posts:
        logger.warning(
            "0 posts returned — either nothing matched the pre-filters, "
            "you haven't joined the channels, or the pre-filters are too "
            "narrow. Try emptying `role_filter` in tg_channels.yaml for "
            "a channel to see all of its posts."
        )
        return

    # === JSON dump ========================================================
    payload = [
        {
            **p.as_vacancy_dict(),
            "preview": p.text[:200].replace("\n", " "),
        }
        for p in posts
    ]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("JSON dump: %s (%d posts)", OUT_JSON.name, len(posts))

    # === Markdown report ==================================================
    lines: list[str] = [
        f"# Telegram pull test — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"- Window: last {HOURS_BACK} hours",
        f"- Channels scanned: {len(channels)}",
        f"- Posts after pre-filter: {len(posts)}",
        "",
        "## Per-channel breakdown",
        "",
    ]

    by_channel: dict[str, list] = {}
    for p in posts:
        by_channel.setdefault(p.channel, []).append(p)

    for ch_name in sorted(by_channel.keys()):
        ch_posts = by_channel[ch_name]
        lines.append(f"- **@{ch_name}** — {len(ch_posts)} post(s)")
    lines.append("")

    lines.append("## Posts")
    lines.append("")
    for p in posts:
        lines.append(f"### @{p.channel} · {p.posted_at.isoformat(timespec='minutes')}")
        lines.append(f"[{p.url}]({p.url})")
        lines.append("")
        # Preview only first 800 chars to keep the report scannable.
        preview = p.text[:800]
        if len(p.text) > 800:
            preview += "\n\n_[truncated, full text in JSON]_"
        lines.append("```")
        lines.append(preview)
        lines.append("```")
        lines.append("")

    OUT_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report: %s", OUT_MARKDOWN.name)

    # === Stdout summary ===================================================
    print()
    print("=" * 60)
    print(f"  Pulled {len(posts)} posts from {len(by_channel)} channel(s)")
    print("=" * 60)
    for ch_name, ch_posts in sorted(by_channel.items()):
        print(f"  @{ch_name:30s} {len(ch_posts):3d} post(s)")
    print()
    print(f"Full JSON:     {OUT_JSON}")
    print(f"Readable MD:   {OUT_MARKDOWN}")
    print()
    print("Next: open the .md file and look at what passed the filter.")
    print("Tune role_filter per channel in data/tg_channels.yaml if needed.")


if __name__ == "__main__":
    asyncio.run(main())
