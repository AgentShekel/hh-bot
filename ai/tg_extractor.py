"""LLM extraction of structured vacancy fields from a raw Telegram post.

Telegram channel posts are free-form text, unlike hh.ru's structured
vacancies. Before the existing `ai/analyzer.py` can score a post we need
to turn it into the `{title, company, salary, ...}` shape it expects —
and, crucially, decide whether the post is even a job opening at all
(channels are full of news, ads, digests and "ищу проект" freelancer
offers).

This runs the CHEAP model tier (flash) with forced JSON output. It is a
pre-gate: only posts it marks `is_vacancy=true` reach the heavier
relevance analyzer + cover-letter pipeline.

Known v1 limitation: multi-vacancy digest posts are reported with
`multi=true` and only the most prominent role is extracted into `title`.
The full post text is still kept as the description and the t.me link
shows everything, so nothing is lost for the user — but the relevance
score is computed against the primary role only. Revisit (split into N
vacancies) if digests turn out to dominate a channel.
"""
from __future__ import annotations

import json
import logging
import re

from ai.llm_client import llm_chat

logger = logging.getLogger(__name__)

# Posts longer than this are truncated before extraction — keeps token
# cost bounded. Job posts rarely carry useful signal past ~4k chars.
_MAX_POST_CHARS = 4000

_SYSTEM_PROMPT = """Ты извлекаешь структуру вакансии из поста Telegram-канала о работе.
Каналы публикуют РАЗНОЕ: вакансии, новости, рекламу, дайджесты, посты
фрилансеров "ищу проект", мемы. Твоя задача — определить, что это, и если
это вакансия от работодателя, вытащить поля.

Верни СТРОГО JSON по схеме (без markdown, без текста вокруг):
{
  "is_vacancy": true|false,
  "multi": true|false,
  "title": "должность одной строкой, либо пустая строка",
  "company": "название компании-работодателя, либо пустая строка",
  "salary": "как написано в посте (300 000-400 000 ₽ / $5k / по договорённости), либо пустая строка",
  "location": "город / Remote / Москва / гибрид, либо пустая строка",
  "is_remote": true|false|null,
  "apply_contact": "как откликаться: @username, email, ссылка, 'в личку автору', 'в комментариях', либо пустая строка",
  "lang": "ru|en"
}

ПРАВИЛА:
- is_vacancy=true ТОЛЬКО если работодатель (или его рекрутер) нанимает на
  конкретную позицию. Реклама курсов, новости индустрии, посты "ищу
  команду/проект" от исполнителя, опросы, мемы — is_vacancy=false и
  остальные поля пустые/null.
- multi=true, если в одном посте НЕСКОЛЬКО разных вакансий (дайджест). В
  этом случае извлеки САМУЮ заметную/первую: её title и поля. Остальные не
  перечисляй.
- title — это роль ("Product Manager", "Senior Developer", "Data Analyst"), без
  компании и города. Не выдумывай, если роли в посте нет — пустая строка.
- apply_contact — самое важное поле. Найди, КАК автор просит откликаться:
  @ник, почта, ссылка на форму/бота, "пишите в личку", "отклик в
  комментариях". Если явно не сказано — пустая строка.
- is_remote: true если удалёнка/remote/из любой точки; false если только
  офис/гибрид с обязательным присутствием; null если не указано.
- Не добавляй полей сверх схемы. Не пиши комментарии. Только JSON."""


def _empty_result(is_vacancy: bool = False) -> dict:
    return {
        "ok": True,
        "is_vacancy": is_vacancy,
        "multi": False,
        "title": "",
        "company": "",
        "salary": "",
        "location": "",
        "is_remote": None,
        "apply_contact": "",
        "lang": "ru",
    }


async def extract_vacancy(post_text: str) -> dict:
    """Extract structured fields from one Telegram post.

    Always returns a dict — never raises (the monitoring loop must keep
    going). Shape:
      {"ok": True, "is_vacancy": bool, "title": str, "company": str,
       "salary": str, "location": str, "is_remote": bool|None,
       "apply_contact": str, "multi": bool, "lang": str}
      {"ok": False}  on LLM/parse failure — caller should retry later
                     (do NOT mark the post seen) rather than drop it.
    """
    text = (post_text or "").strip()
    if not text:
        return _empty_result(is_vacancy=False)
    if len(text) > _MAX_POST_CHARS:
        text = text[:_MAX_POST_CHARS] + "..."

    try:
        raw = await llm_chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Пост канала:\n\"\"\"\n{text}\n\"\"\""},
            ],
            temperature=0.1,
            max_tokens=600,
            model_tier="standard",
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning("tg_extractor LLM failed: %s", e)
        return {"ok": False}

    parsed = _parse_json(raw)
    if parsed is None:
        logger.warning("tg_extractor JSON parse failed: %r", (raw or "")[:200])
        return {"ok": False}

    # Normalise into a complete, well-typed result.
    result = _empty_result(is_vacancy=bool(parsed.get("is_vacancy")))
    result["multi"] = bool(parsed.get("multi"))
    for key in ("title", "company", "salary", "location", "apply_contact", "lang"):
        val = parsed.get(key)
        result[key] = val.strip() if isinstance(val, str) else ""
    remote = parsed.get("is_remote")
    result["is_remote"] = remote if isinstance(remote, bool) else None
    if not result["lang"]:
        result["lang"] = "ru"
    return result


def _parse_json(raw: str) -> dict | None:
    """Best-effort JSON extraction from the model output."""
    if not raw:
        return None
    text = raw.strip()
    # Strip a ```json ... ``` fence if the model added one despite instructions.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
