"""One-shot: process today's 80+ vacancies without responses, using current
cover-letter prompt. Runs independently from autopilot.

Older vacancies (>3 days) are usually archived on hh.ru and have no apply
button. We filter to recent only to avoid wasted attempts.
"""
import asyncio
import logging
import sys
import sqlite3
from parser.hh_client import HHClient
from ai.cover_letter import generate_cover_letter
from db.storage import save_response, log_action
from config import DB_PATH

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("process_pending")


def get_recent_candidates(days: int = 3) -> list[dict]:
    """80+ vacancies without response, found in last N days."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT v.* FROM vacancies v
               LEFT JOIN responses r ON v.id = r.vacancy_id
               WHERE r.id IS NULL
                 AND v.relevance_score >= 80
                 AND v.found_at >= datetime('now', ?)
               ORDER BY v.relevance_score DESC""",
            (f'-{days} days',),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


async def main():
    candidates = get_recent_candidates(days=3)
    print(f"Found {len(candidates)} recent candidates (last 3 days, score >= 80, no response)")
    if not candidates:
        return

    for v in candidates:
        print(f"  - {v['relevance_score']:>3}  {v['title'][:60]}  ({v['company']})")

    print()
    print("Starting hh client...")
    hh = HHClient()
    status = await hh.start()
    print(f"  status: {status}")

    if status != "session_ok":
        print(f"  FAIL: status is {status}, cannot proceed (need_login? run main.py and /login first)")
        await hh.stop()
        return

    for v in candidates:
        try:
            print(f"\n>>> {v['title'][:60]}")
            letter = await generate_cover_letter(v)
            if not letter:
                print("    SKIP: empty cover letter")
                continue
            print(f"    letter ({len(letter)} chars): {letter[:120]}...")
            success = await hh.apply_to_vacancy(v["id"], letter)
            if success:
                save_response(v["id"], letter, "sent")
                log_action("auto_applied", f"vacancy={v['id']} (manual_runner)")
                print(f"    OK: applied")
            else:
                print(f"    FAIL: apply returned False")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")

    await hh.stop()
    print("\nDone.")


asyncio.run(main())
