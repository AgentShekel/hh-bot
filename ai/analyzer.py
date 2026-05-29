"""Vacancy relevance analyzer.

Loads candidate summary from `prompts/analyzer_summary.txt` (local, gitignored).
Falls back to `prompts/analyzer_summary.example.txt` for fresh clones.
"""
import json
import logging
import re
from pathlib import Path

from ai.llm_client import llm_chat

logger = logging.getLogger(__name__)


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SUMMARY_PATH = _PROMPTS_DIR / "analyzer_summary.txt"
_SUMMARY_EXAMPLE_PATH = _PROMPTS_DIR / "analyzer_summary.example.txt"


def _load_candidate_summary() -> str:
    if _SUMMARY_PATH.exists():
        return _SUMMARY_PATH.read_text(encoding="utf-8").strip()
    if _SUMMARY_EXAMPLE_PATH.exists():
        logger.warning(
            "Using prompts/analyzer_summary.example.txt. Copy it to "
            "prompts/analyzer_summary.txt and fill in your data."
        )
        return _SUMMARY_EXAMPLE_PATH.read_text(encoding="utf-8").strip()
    logger.error("No analyzer summary found in %s.", _PROMPTS_DIR)
    return ""


# ──────────────────────────────────────────────────────────────────────────
# HOW TO TUNE SCORING TO YOUR PROFILE (read this if you forked the project)
# ──────────────────────────────────────────────────────────────────────────
# This analyzer is intentionally GENERIC. It scores how well a vacancy fits
# the candidate described in `prompts/analyzer_summary.txt` (your data — see
# the .example template). Step 1 is just to fill that summary in.
#
# The prompt below gives the model a neutral scoring framework: parse the real
# role, weigh must-have vs nice-to-have, compare against the candidate's target
# roles / strengths / gaps / work-format from the summary, output 0-100.
#
# If you want STRICTER, profile-specific scoring, extend the "SCORING
# PRINCIPLES" block in `_build_prompt` with your own rules, e.g.:
#   • hard role-type mismatch list (professions to auto-downrate to 5-15);
#   • anti-domain list (industries you never apply to → cap at 30);
#   • base scores per seniority / per target-title;
#   • hard work-format filter (e.g. remote-only → cap on-site at 10).
# Keep specifics OUT of the public summary if you share this repo — put them
# here as your own tuning, or in your private analyzer_summary.txt.
# ──────────────────────────────────────────────────────────────────────────


def _build_prompt(title: str, company: str, description: str) -> str:
    return f"""Ты — senior tech-рекрутер. Оцени, насколько вакансия подходит
кандидату, чей профиль дан ниже (раздел «Кандидат»). Не суди поверхностно по
одному title — разбери суть роли.

КАК ОЦЕНИВАТЬ:
1. Разбери ОБЯЗАННОСТИ — что РЕАЛЬНО предстоит делать (раздел обязанности /
   задачи / responsibilities), а не лозунг из шапки.
2. Раздели требования на must-have и "будет плюсом".
3. Определи role-type (к какой профессии реально относится роль) и уровень.
4. Сопоставь с профилем кандидата из summary: целевые роли, сильные стороны,
   чего у него НЕТ, желаемый формат работы.
5. Выстави балл 0-100 соответствия и верни JSON (схема в конце).

ПРИНЦИПЫ СКОРИНГА (общие; всю конкретику бери из профиля кандидата в summary):
- Прямое попадание в ЦЕЛЕВЫЕ роли кандидата (из summary) → высокий балл (75-95).
- Смежные или более общие роли → средний балл (40-65).
- Роль из ДРУГОЙ профессии (не из целевых кандидата) → низкий балл (5-20),
  даже если в описании совпадают отдельные ключевые слова. Реши по СУТИ
  обязанностей, а не по одному слову в title.
- Если вакансия требует как MUST-HAVE навык/опыт, которого у кандидата НЕТ
  (см. раздел gaps в summary) — снижай балл тем сильнее, чем центральнее этот
  навык для роли. Формулировка "будет плюсом" / "nice to have" — НЕ снижает.
- Учитывай формат работы и уровень, заявленные кандидатом в summary (если он
  ограничил — напр. только удалёнка или определённый уровень — несоответствие
  понижает балл).
- Если в JD есть прямая инструкция к письму («ответьте на вопросы N»,
  «кодовое слово», «опишите проект в письме») — НЕ понижай балл, но отметь
  в reason: «требует custom cover letter».

Кандидат: {_load_candidate_summary()}
Вакансия: {title} в {company}. {description}

Верни СТРОГО JSON по схеме (без markdown, без текста вокруг, только объект):
{{
  "role_type": "к какой профессии реально относится роль",
  "ai_focus": "high|medium|low",
  "role_level": "junior|middle|senior|lead|head|director|неясно",
  "remote": "да|нет|неясно",
  "key_match": "сильнейшее совпадение роли с профилем кандидата, одна фраза",
  "key_gap": "главный гэп или red flag для этой роли, либо 'нет'",
  "relevance": "высокая|средняя|низкая",
  "relevance_score": 0,
  "reason": "2-3 предложения с конкретикой из вакансии: чем обоснован балл — role-type, ключевое совпадение, и решающий гэп/red flag если есть. Без общих слов."
}}

relevance_score — целое 0-100 по принципам выше. reason должен объяснять
ИМЕННО этот балл, со ссылкой на конкретные пункты вакансии."""


async def analyze_relevance(vacancy: dict, deep: bool = True) -> dict:
    """Analyze vacancy relevance. Returns dict with relevance and score.

    deep=True  -> pro model with reasoning (hh.ru autopilot: precise, drives
                  auto-apply decisions).
    deep=False -> cheap flash model (Telegram monitor: light, lenient pass —
                  TG is a discovery feed, we'd rather surface than over-cut).
    """
    title = vacancy.get("title", "")
    company = vacancy.get("company", "")
    description = vacancy.get("description", "")

    if len(description) > 3000:
        description = description[:3000] + "..."

    prompt = _build_prompt(title, company, description)

    try:
        text = await llm_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
            model_tier="premium" if deep else "standard",
            response_format={"type": "json_object"},
        )
        logger.info("LLM raw (first 200): %s", text[:200])
        result = _parse_relevance(text)
        if result["relevance"] == "unknown" and not result["reason"]:
            result["reason"] = text[:160] if text else "empty response"
        return result
    except Exception as e:
        logger.error("LLM relevance error: %s", e)
        return {"relevance": "unknown", "relevance_score": 0, "reason": str(e)}


def _normalize_relevance(val: str) -> str:
    v = (val or "").lower()
    if "высок" in v or "high" in v:
        return "high"
    if "средн" in v or "medium" in v:
        return "medium"
    if "низк" in v or "low" in v:
        return "low"
    return "unknown"


def _try_json(text: str) -> dict | None:
    """Extract a JSON object from the model output, or None."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, flags=re.DOTALL)
    if m:
        t = m.group(1)
    else:
        first, last = t.find("{"), t.rfind("}")
        if first >= 0 and last > first:
            t = t[first : last + 1]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_relevance(text: str) -> dict:
    # Preferred path: structured JSON from the analyzer.
    parsed = _try_json(text)
    if parsed is not None and (
        "relevance_score" in parsed or "relevance" in parsed
    ):
        try:
            score = int(parsed.get("relevance_score", 0))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))
        # Derive the relevance label from the score so the card emoji matches
        # the number (the model sometimes says "высокая" while scoring 40).
        # Fall back to the model's own label only when there's no usable score.
        if score >= 70:
            relevance = "high"
        elif score >= 40:
            relevance = "medium"
        elif score > 0:
            relevance = "low"
        else:
            relevance = _normalize_relevance(str(parsed.get("relevance", "")))
        reason = str(parsed.get("reason") or "").strip()
        out = {"relevance": relevance, "relevance_score": score, "reason": reason}
        # Carry structured extras — harmless to downstream (save_vacancy
        # ignores unknown keys), useful in logs and for richer cards later.
        for k in ("role_type", "ai_focus", "role_level", "key_match", "key_gap"):
            if parsed.get(k):
                out[k] = parsed[k]
        return out

    # Fallback: legacy 3-line text format (РЕЛЕВАНТНОСТЬ/БАЛЛ/ПРИЧИНА).
    return _parse_relevance_text(text)


def _parse_relevance_text(text: str) -> dict:
    relevance = "unknown"
    score = 0
    reason = ""

    lower_text = text.lower()

    # parse relevance
    for line in text.split("\n"):
        lower = line.strip().lower()
        if "релевантность" in lower:
            val = line.split(":", 1)[-1].strip().lower() if ":" in line else lower
            if "высок" in val or "high" in val:
                relevance = "high"
            elif "средн" in val or "medium" in val:
                relevance = "medium"
            elif "низк" in val or "low" in val:
                relevance = "low"
            break

    # fallback
    if relevance == "unknown":
        if "высок" in lower_text:
            relevance = "high"
        elif "средн" in lower_text:
            relevance = "medium"
        elif "низк" in lower_text:
            relevance = "low"

    # parse score
    for line in text.split("\n"):
        lower = line.strip().lower()
        if "балл" in lower or "score" in lower:
            digits = re.findall(r"\d+", line)
            if digits:
                val = int(digits[0])
                if 0 <= val <= 100:
                    score = val
                    break

    if score == 0:
        matches = re.findall(r"(\d{1,3})\s*/?\s*100", text)
        if matches:
            score = int(matches[0])

    # parse reason
    for line in text.split("\n"):
        lower = line.strip().lower()
        if "причина" in lower or "reason" in lower:
            reason = line.split(":", 1)[-1].strip() if ":" in line else ""
            break

    return {"relevance": relevance, "relevance_score": score, "reason": reason}
