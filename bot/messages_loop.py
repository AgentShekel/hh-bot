"""Periodically check hh.ru inbox for new employer messages.

Notifies Telegram on new messages, flags likely test-task messages with
a special tag so the user can react fast.
"""
import asyncio
import hashlib
import logging

from aiogram import Bot

from config import TELEGRAM_ADMIN_ID
from db.storage import is_message_seen, save_message, log_action
from parser.hh_client import HHClient
from bot.autopilot import HH_LOCK

logger = logging.getLogger(__name__)

MESSAGES_INTERVAL = 5 * 60  # 5 min between inbox sweeps
INITIAL_DELAY = 90  # let autopilot start first

# Heuristic keywords for test-task / homework messages.
TEST_TASK_KEYWORDS = [
    "тестовое",
    "тестов задани",
    "test task",
    " тз ",
    " тз.",
    "задание",
    "выполните",
    "выполнить задание",
    "решите",
    "assignment",
    "homework",
    "кейс",
]

# Heuristic keywords for questions / requests requiring an answer.
QUESTION_KEYWORDS = [
    "вопрос",
    "уточни",
    "расскажи",
    "пожалуйста, опиши",
    "ответьте",
    "можете ли",
    "?",
]


def _classify(preview: str) -> str:
    """Return tag based on message content."""
    p = preview.lower()
    if any(kw in p for kw in TEST_TASK_KEYWORDS):
        return "TEST_TASK"
    if any(kw in p for kw in QUESTION_KEYWORDS):
        return "QUESTION"
    return "MESSAGE"


def _hash_preview(preview: str) -> str:
    return hashlib.md5(preview.strip().encode("utf-8")).hexdigest()[:16]


async def _fetch_threads(hh_client: HHClient) -> list[dict]:
    """Single inbox fetch under the browser lock."""
    async with HH_LOCK:
        if not await hh_client._is_logged_in():
            logger.info("Messages loop: not logged in, skipping sweep")
            return []
        return await hh_client.get_unread_messages()


async def _notify_thread(bot: Bot, thread: dict) -> None:
    preview = (thread.get("preview") or "").strip()
    if not preview:
        return
    content_hash = _hash_preview(preview)
    chat_url = thread.get("chat_url") or ""

    if is_message_seen(chat_url, content_hash):
        return

    tag = _classify(preview)
    is_test = tag == "TEST_TASK"

    save_message(
        chat_url=chat_url,
        content_hash=content_hash,
        vacancy_id=thread.get("vacancy_id", ""),
        vacancy_title=thread.get("vacancy_title", ""),
        employer=thread.get("employer", ""),
        preview=preview,
        is_test_task=is_test,
    )

    tag_label = {
        "TEST_TASK": "[TEST TASK]",
        "QUESTION": "[QUESTION]",
        "MESSAGE": "[NEW MSG]",
    }[tag]

    text_parts = [
        f"{tag_label} новое сообщение от работодателя",
        "",
        f"Вакансия: {thread.get('vacancy_title') or '?'}",
        f"Работодатель: {thread.get('employer') or '?'}",
        "",
        preview[:1000],
    ]
    if chat_url:
        text_parts += ["", chat_url]
    text = "\n".join(text_parts)

    try:
        await bot.send_message(TELEGRAM_ADMIN_ID, text)
        log_action(
            "inbox_notify",
            f"tag={tag} vacancy={thread.get('vacancy_title', '')}",
        )
    except Exception as e:
        logger.error("Failed to send TG inbox notification: %s", e)


async def messages_loop(bot: Bot, hh_client: HHClient):
    """Sweep hh.ru inbox every MESSAGES_INTERVAL and push new messages to Telegram."""
    await asyncio.sleep(INITIAL_DELAY)
    logger.info("Messages loop started")

    while True:
        try:
            threads = await _fetch_threads(hh_client)
        except Exception as e:
            logger.error("Messages loop fetch error: %s", e)
            threads = []

        for t in threads:
            try:
                await _notify_thread(bot, t)
            except Exception as e:
                logger.error("Messages loop notify error: %s", e)

        await asyncio.sleep(MESSAGES_INTERVAL)
