"""Autopilot: continuous search + scoring + routing. Auto-apply is opt-in
(AUTO_APPLY_ENABLED); scores below SCORE_AUTO_SKIP are dropped; everything
else goes to Telegram for manual review. Posts a periodic summary."""
import asyncio
import logging
import os
from datetime import datetime

from aiogram import Bot

from config import TELEGRAM_ADMIN_ID, AUTO_APPLY_ENABLED
from db.storage import (
    save_vacancy,
    save_response,
    mark_skipped,
    get_active_filters,
    get_auto_apply_candidates,
    get_employer_rating_cached,
    save_employer_rating,
    is_duplicate_already_handled,
    log_action,
)
from parser.hh_client import HHClient
from ai.analyzer import analyze_relevance
from ai.cover_letter import generate_cover_letter

logger = logging.getLogger(__name__)

# Shared lock for hh.ru browser access — autopilot and messages_loop both
# need self.page; serialise to avoid clobbering navigation.
HH_LOCK = asyncio.Lock()

# Telegram summary cadence (env-overridable). Empty windows are skipped in
# send_summary (no spam).
SUMMARY_INTERVAL = int(os.getenv("AUTOPILOT_SUMMARY_INTERVAL", str(30 * 60)))
# Pacing knobs are env-overridable so you can dial volume down via .env without
# touching code. If hh.ru starts returning short "you're not a robot" anti-bot
# pages (watch for "short body" warnings in the log), increase the pauses —
# a slower pace avoids the proxy-IP rate limit.
SEARCH_PAUSE = int(os.getenv("AUTOPILOT_SEARCH_PAUSE", str(30 * 60)))  # default: 30 min

# === Scoring thresholds (0-100 relevance from the analyzer) ===
# SCORE_AUTO_APPLY: at or above this the autopilot may auto-apply (only when
#   AUTO_APPLY_ENABLED=true). Keep it high so only very confident hits apply.
# SCORE_AUTO_SKIP: below this the vacancy is auto-skipped as a clear mismatch.
# Everything in between goes to Telegram for manual review. Tune both to your
# analyzer's behaviour and how noisy a review queue you tolerate.
# SCORE_AUTO_SKIP is imported by bot/tg_loop.py too (same floor for the TG monitor).
SCORE_AUTO_APPLY = 90
SCORE_AUTO_SKIP = 10

MAX_VACANCIES_PER_CYCLE = int(os.getenv("AUTOPILOT_MAX_VACANCIES_PER_CYCLE", "20"))  # per filter
VACANCY_PAUSE = int(os.getenv("AUTOPILOT_VACANCY_PAUSE", "90"))  # seconds between vacancy analysis

# === Profile-overlap rescue ===
# Even when the analyzer scores a vacancy below SCORE_AUTO_SKIP, the JD may
# strongly overlap the candidate's actual profile (see PROFILE_MARKERS below).
# Rescue such vacancies to manual review with a tag showing which marker
# categories matched.
#
# Each category contributes its weight AT MOST ONCE — multiple hits in
# one category don't compound (defends against "10x Python in one JD"
# noise). Threshold 3 means: any AI/LLM-core or AI-product match alone
# triggers, OR a combo of Product/PM + Tech-stack, OR Product/PM + Domain.
# Pure tech-stack-only (Python dev) or pure-domain-only (B2B SaaS jobpost)
# do NOT trigger — those are usually mismatches the analyzer rightly cut.
PROFILE_OVERLAP_THRESHOLD = 3
# ── TUNE ME ────────────────────────────────────────────────────────────────
# PROFILE_MARKERS below is an ILLUSTRATIVE EXAMPLE for an AI/product-oriented
# job seeker; the keywords are generic placeholders. Replace each category's
# list with the signal words of YOUR OWN profile before relying on the rescue.
# Keep the category KEYS (especially "product/pm" and "web/agency stack"):
# they are referenced by the overlap-rescue branch in run_search_cycle() and in
# bot/tg_loop.py + scripts/overlap_rescue_existing.py. If you rename or drop a
# category, update those call sites too. Markers work best when SPECIFIC;
# overly generic tokens over-trigger on long JD stack enumerations.
# ───────────────────────────────────────────────────────────────────────────
PROFILE_MARKERS: dict[str, tuple[int, list[str]]] = {
    "ai/llm core": (3, [
        " llm", "large language model", "agentic", " ai agent",
        "multi-agent", "мультиагент",
        "prompt engineering", "промпт-инжиниринг",
        "openai api", "claude api",
        "rag", "fine-tun", " lora",
        "genai", "gen ai", "gen-ai",
        "evaluation pipeline", "human-in-the-loop",
        "vector db", "embedding",
    ]),
    "ai product": (3, [
        "ai product", "ai-product",
        "ai implementation", "ai-implementation",
        "ai transformation", "ai-transformation",
        "ai adoption", "ai-adoption",
        "ai strategy", "ai-strategy",
        "head of ai", "ai-внедрен", "внедрение ai",
    ]),
    "product/pm": (2, [
        # Product manager titles
        "product manager", "product owner", "product lead",
        "head of product", "руководитель продукта",
        # Generic management / lead titles. These ride on title-aware overlap
        # (the title is included in the searched text at the call site) and
        # separate a "manager who codes" from a pure IC developer.
        "team lead", "tech lead", "engineering manager",
        "head of ", "director of",
        "руководитель отдела", "руководитель направления",
        "руководитель группы", "руководитель проект",
        "директор по ", "технический руководитель",
        # PM activities (signal even without a management title)
        "roadmap", "a/b test", "ab-тест",
        "stakeholder", "стейкхолдер",
        " kpi", " okr",
    ]),
    # EXAMPLE backend/frontend stack — swap for the one on your résumé.
    # Keep tokens specific enough that they don't fire on every JD that lists
    # the tech in a long enumeration.
    "tech stack": (2, [
        "fastapi", "django", "asyncio",
        "postgres", "redis", "docker",
        "react", "typescript",
    ]),
    # OPTIONAL second profile — EXAMPLE only. Use this if you have a separate
    # skill set worth surfacing that isn't on your main résumé (a different
    # stack or domain). Otherwise tune the keywords or drop the category (and
    # its references in the call sites named above). Keep markers SPECIFIC and
    # trigger only when COMBINED with another category.
    "web/agency stack": (2, [
        "wordpress", "drupal", "joomla",
        "laravel", "symfony",
        "cms-сайт", "разработ под cms",
        "адаптивн вёрстк", "адаптивн верстк",
    ]),
    "domain": (1, [
        "b2b saas", "on-premise", "on premise",
        "ecommerce", "e-commerce", "marketplace",
        "fintech", "edtech",
    ]),
}


# Title-level engineer/designer markers — these indicate the role is
# an IC engineer/designer position, not a management/PM role. Used by
# `_title_looks_like_ic_role()` to suppress overlap-rescue when title
# is clearly not the candidate's profile, regardless of how rich the
# JD body is in AI/PM keywords.
_IC_ROLE_TITLE_MARKERS = [
    # English — always preceded by a space (no compound words)
    " developer", " engineer", " designer", " analyst",
    " researcher", " specialist", " programmer",
    # Russian — no leading space so hyphenated forms also match
    # ("Backend-разработчик", "ML-инженер", "Frontend-разработчик")
    "разработчик", "программист",
    "инженер", "дизайнер",
    "аналитик", "исследовател", "специалист",
    "архитектор",  # IC architect roles
]

# Title-level management anchors — these "rescue" titles that contain an
# IC marker but ALSO indicate management (e.g. "Engineering Manager",
# "Tech Lead Backend"). If present, overlap-rescue is allowed.
_MANAGEMENT_TITLE_ANCHORS = [
    " manager", "head of", "руководитель", "директор по",
    "team lead", "tech lead", "engineering lead", "engineering manager",
    "chief ", " cto", " cpo", " cdo", " vp ",
    "product manager", "product owner", "product lead",
]


def _title_looks_like_ic_role(title: str) -> bool:
    """True if title is a clear IC engineer/designer/analyst role
    WITHOUT a management anchor. When the configured profile targets
    management / product roles, pure IC roles shouldn't be rescued even when
    the JD body is rich in AI/PM keywords (description-keyword false positives).
    """
    if not title:
        return False
    t = title.lower()
    has_ic_marker = any(m in t for m in _IC_ROLE_TITLE_MARKERS)
    if not has_ic_marker:
        return False
    has_mgmt = any(a in t for a in _MANAGEMENT_TITLE_ANCHORS)
    return not has_mgmt


def profile_overlap_score(
    description: str, title: str = ""
) -> tuple[int, list[str]]:
    """Score how strongly a JD overlaps the configured candidate profile.

    Searches both `title` and `description` — titles often carry the
    cleanest specificity ("Team Lead", "Product Manager"), while
    descriptions can dilute signals in long stack enumerations.

    Returns (total, matched_categories). Each category is all-or-nothing
    (its weight applied once if any of its markers occurs in either
    field), so noisy JDs with the same keyword repeated don't inflate
    the score. Used by the autopilot to "rescue" low-LLM-score
    vacancies into manual review — see PROFILE_OVERLAP_THRESHOLD.
    """
    if not description and not title:
        return 0, []
    text = f"{title or ''} {description or ''}".lower()
    total = 0
    matched: list[str] = []
    for cat, (weight, markers) in PROFILE_MARKERS.items():
        if any(m in text for m in markers):
            total += weight
            matched.append(cat)
    return total, matched

# Minimum employer rating on hh.ru to consider applying. Companies without
# a rating widget (None) are NOT filtered out — small / new companies often
# have no rating yet, and some of them are AI startups worth applying to.
MIN_COMPANY_RATING = 3.5

# Blacklisted companies — never apply (whole-word match on company name).
# Empty by default. Add employers you never want to apply to, e.g.:
#   COMPANY_BLACKLIST = ["acme corp", "example ltd"]
COMPANY_BLACKLIST: list[str] = []

# Title pre-filter — substring match on the vacancy title (lowercased), runs
# BEFORE the LLM analyzer to save tokens and act as a hard floor against
# analyzer mistakes. EMPTY by default — nothing is pre-filtered, every vacancy
# reaches the analyzer.
#
# TUNE ME: add lowercase title fragments of roles/domains that are DEFINITELY
# NOT for you, so they're dropped without an LLM call. This is YOUR personal
# anti-list — what counts as "off-target" depends entirely on your profile
# (a designer would NOT blacklist "designer"; a salesperson would NOT
# blacklist "sales"). Example for someone who wants to skip a few unrelated
# role types:
#   TITLE_BLACKLIST = ["3d artist", "sales manager", "recruiter", "бухгалтер"]
# Matched against the title only (not the description), so false negatives are
# preferred over false positives.
TITLE_BLACKLIST: list[str] = []

# accumulate stats between summaries
_stats = {
    "found": 0,
    "auto_applied": [],
    "manual_review": 0,
    "auto_skipped": 0,
    "errors": 0,
    "last_summary": datetime.now(),
}


def _reset_stats():
    _stats["found"] = 0
    _stats["auto_applied"] = []
    _stats["manual_review"] = 0
    _stats["auto_skipped"] = 0
    _stats["errors"] = 0
    _stats["last_summary"] = datetime.now()


async def autopilot_loop(bot: Bot, hh_client: HHClient):
    """Main autopilot loop. Searches continuously, sends summary every 2 hours."""
    await asyncio.sleep(60)  # wait 1 min after startup
    logger.info("Autopilot started")

    # start summary task
    asyncio.create_task(summary_loop(bot))

    while True:
        try:
            async with HH_LOCK:
                logged_in = await hh_client._is_logged_in()
            if not logged_in:
                logger.warning("Autopilot: not logged in, sleeping 10 min")
                await asyncio.sleep(SEARCH_PAUSE)
                continue

            await run_search_cycle(bot, hh_client)

        except Exception as e:
            logger.error("Autopilot error: %s", e)
            _stats["errors"] += 1
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning("Autopilot: rate limited, extra 10 min cooldown")
                await asyncio.sleep(600)

        await asyncio.sleep(SEARCH_PAUSE)


async def summary_loop(bot: Bot):
    """Send summary every 2 hours."""
    while True:
        await asyncio.sleep(SUMMARY_INTERVAL)
        try:
            await send_summary(bot)
        except Exception as e:
            logger.error("Summary send error: %s", e)


async def run_search_cycle(bot: Bot, hh_client: HHClient):
    """Run one search cycle: find, analyze, auto-apply/skip."""
    filters = get_active_filters()
    if not filters:
        return

    for f in filters:
        try:
            async with HH_LOCK:
                vacancies = await hh_client.search_vacancies(f)
            processed = 0
            for v in vacancies:
                if processed >= MAX_VACANCIES_PER_CYCLE:
                    logger.info("Autopilot: reached %d vacancy limit for this cycle", MAX_VACANCIES_PER_CYCLE)
                    break
                # Check company blacklist
                company_lower = v.get("company", "").lower()
                if any(bl in company_lower for bl in COMPANY_BLACKLIST):
                    logger.info("Autopilot: skipping %s (blacklisted company: %s)", v["id"], v.get("company"))
                    v["relevance_score"] = 0
                    v["relevance"] = "low"
                    v["reason"] = f"Компания в черном списке: {v.get('company')}"
                    v["description"] = ""
                    save_vacancy(v)
                    mark_skipped(v["id"])
                    _stats["auto_skipped"] += 1
                    continue

                # Check title topic blacklist (off-target title fragments, if configured)
                title_lower = v.get("title", "").lower()
                blacklisted_topic = next(
                    (bl for bl in TITLE_BLACKLIST if bl in title_lower), None
                )
                if blacklisted_topic:
                    logger.info("Autopilot: skipping %s (title topic: %s) — %s",
                                v["id"], blacklisted_topic, v.get("title"))
                    v["relevance_score"] = 0
                    v["relevance"] = "low"
                    v["reason"] = f"Топик вне профиля: {blacklisted_topic}"
                    v["description"] = ""
                    save_vacancy(v)
                    mark_skipped(v["id"])
                    _stats["auto_skipped"] += 1
                    continue

                # Cross-city / cross-format duplicate check: same employer
                # + same normalized title already handled before.
                if is_duplicate_already_handled(v.get("company", ""), v.get("title", "")):
                    logger.info(
                        "Autopilot: skipping %s (duplicate of already-handled "
                        "vacancy from %s) — %s",
                        v["id"], v.get("company"), v.get("title"),
                    )
                    v["relevance_score"] = 0
                    v["relevance"] = "low"
                    v["reason"] = "Дубликат вакансии того же работодателя (уже обработана)"
                    v["description"] = ""
                    save_vacancy(v)
                    mark_skipped(v["id"])
                    _stats["auto_skipped"] += 1
                    continue

                async with HH_LOCK:
                    desc = await hh_client.get_vacancy_description(v["url"])
                    v["description"] = desc
                    # check remote + rating use the just-loaded page, must be under same lock
                    is_remote = await hh_client.check_remote_available()
                    rating = await hh_client.get_company_rating()
                    rating_source = "vacancy" if rating is not None else None
                    # Fallback: try the employer page (with 30-day cache).
                    if rating is None:
                        employer_id = await hh_client.get_employer_id_from_vacancy_page()
                        if employer_id:
                            cached_found, cached_rating = get_employer_rating_cached(employer_id)
                            if cached_found:
                                rating = cached_rating
                                rating_source = "cache"
                            else:
                                rating = await hh_client.fetch_employer_rating(employer_id)
                                save_employer_rating(employer_id, rating)
                                rating_source = "employer-page"
                v["company_rating"] = rating or 0
                logger.info(
                    "Vacancy %s | %s | remote=%s | rating=%s (%s)",
                    v["id"], v.get("company", "?")[:40], is_remote,
                    f"{rating:.1f}" if rating is not None else "n/a",
                    rating_source or "none",
                )

                if not is_remote:
                    logger.info("Autopilot: skipping %s (no remote), %s", v["id"], v["title"])
                    v["relevance_score"] = 0
                    v["relevance"] = "low"
                    v["reason"] = "Нет удалённой работы"
                    save_vacancy(v)
                    mark_skipped(v["id"])
                    _stats["auto_skipped"] += 1
                    await asyncio.sleep(VACANCY_PAUSE)
                    continue

                if rating is not None and rating > 0 and rating < MIN_COMPANY_RATING:
                    logger.info(
                        "Autopilot: skipping %s (low rating %.1f < %.1f), %s",
                        v["id"], rating, MIN_COMPANY_RATING, v["title"],
                    )
                    v["relevance_score"] = 0
                    v["relevance"] = "low"
                    v["reason"] = f"Низкий рейтинг компании: {rating:.1f}/5 (порог {MIN_COMPANY_RATING})"
                    save_vacancy(v)
                    mark_skipped(v["id"])
                    _stats["auto_skipped"] += 1
                    await asyncio.sleep(VACANCY_PAUSE)
                    continue

                # Light flash screening (deep=False) to keep autopilot fast and
                # off the overloaded DeepSeek/proxy path. The deep pro analysis
                # still runs at letter-generation time (generate_cover_letter).
                analysis = await analyze_relevance(v, deep=False)
                v.update(analysis)

                if save_vacancy(v):
                    processed += 1
                    _stats["found"] += 1
                    score = v.get("relevance_score", 0)

                    # Route by score, with a profile-overlap rescue branch
                    # for low-scored vacancies that nevertheless overlap
                    # the candidate's actual profile (see PROFILE_MARKERS).
                    overlap_meta = ""  # populated only on rescue
                    if score < SCORE_AUTO_SKIP:
                        overlap, matched_cats = profile_overlap_score(
                            v.get("description", ""),
                            v.get("title", ""),
                        )
                        # Rescue requires:
                        #   (a) total overlap >= threshold, AND
                        #   (b) at least one ROLE-relevant category —
                        #       "product/pm" (management/lead role) OR
                        #       "web/agency stack" (second-profile web roles), AND
                        #   (c) title is NOT a clear IC engineer/designer
                        #       role without a management anchor.
                        # Why (c): JD bodies of engineer vacancies often
                        # describe team context with management words
                        # ("you'll work with the tech lead", "stakeholder
                        # alignment"), inflating product/pm category for
                        # a role the candidate doesn't take. Title is the
                        # cleanest signal.
                        role_match = (
                            "product/pm" in matched_cats
                            or "web/agency stack" in matched_cats
                        )
                        ic_role = _title_looks_like_ic_role(v.get("title", ""))
                        if (
                            overlap >= PROFILE_OVERLAP_THRESHOLD
                            and role_match
                            and not ic_role
                        ):
                            overlap_meta = (
                                f"[overlap={overlap}] cats: "
                                f"{', '.join(matched_cats)}"
                            )
                            logger.info(
                                "Autopilot: rescuing %s to manual review "
                                "(score=%d, %s) — %s",
                                v["id"], score, overlap_meta, v["title"],
                            )
                            # fall through to the manual-review branch below
                        else:
                            mark_skipped(v["id"])
                            _stats["auto_skipped"] += 1
                            await asyncio.sleep(VACANCY_PAUSE)
                            continue

                    if AUTO_APPLY_ENABLED and score >= SCORE_AUTO_APPLY:
                        success = await auto_apply(hh_client, v)
                        if not success:
                            _stats["errors"] += 1

                    else:
                        # 30-89 OR low-score-but-rescued-by-overlap:
                        # send to Telegram for manual review.
                        _stats["manual_review"] += 1
                        from bot.keyboards import vacancy_keyboard
                        rating_str = (
                            f"Рейтинг компании: {v.get('company_rating'):.1f}/5\n"
                            if v.get("company_rating") else ""
                        )
                        if overlap_meta:
                            # Rescue card: lead with overlap signal, not
                            # with the analyzer's low score (low here
                            # means "analyzer cut by role-type/AI-focus
                            # gate, but JD overlaps your profile" — NOT
                            # "bad vacancy"). Showing 10/100 first
                            # reads as "skip this" and people pass.
                            text = (
                                f"🎯 OVERLAP-RESCUE\n"
                                f"{v['title']}\n"
                                f"{v.get('company', '')}\n"
                                f"Зарплата: {v.get('salary') or 'не указана'}\n"
                                f"Город: {v.get('city') or ''}\n"
                                f"{rating_str}"
                                f"{overlap_meta}\n"
                                f"(analyzer={v.get('relevance_score', 0)}/100 — "
                                f"низкий, режется по role-type гейту; "
                                f"подняли через overlap-маркеры твоего профиля)"
                            )
                        else:
                            relevance_emoji = {"high": "[!!!]", "medium": "[!!]", "low": "[!]"}.get(
                                v.get("relevance", ""), "[?]"
                            )
                            # Auto-apply is frozen — flag would-be auto-apply
                            # hits (90+) as top-priority so you react fast.
                            if v.get("relevance_score", 0) >= SCORE_AUTO_APPLY:
                                relevance_emoji = "🔥 СИЛЬНОЕ"
                            text = (
                                f"{relevance_emoji} {v['title']}\n"
                                f"{v.get('company', '')}\n"
                                f"Зарплата: {v.get('salary') or 'не указана'}\n"
                                f"Город: {v.get('city') or ''}\n"
                                f"{rating_str}"
                                f"Релевантность: {v.get('relevance_score', 0)}/100\n"
                                f"Причина: {v.get('reason', '')}"
                            )
                        await bot.send_message(TELEGRAM_ADMIN_ID, text, reply_markup=vacancy_keyboard(v["id"], v["url"]))

                await asyncio.sleep(VACANCY_PAUSE)

        except Exception as e:
            logger.error("Autopilot search error for filter %s: %s", f["name"], e)
            _stats["errors"] += 1

    # also check previously found 90+ vacancies without response.
    # Capped at MAX_VACANCIES_PER_CYCLE so the per-cycle request volume stays
    # bounded — without this, a large backlog (e.g. 27 score-90 rows) gets
    # hammered in one cycle, which on a flagged/slow proxy IP both crawls and
    # feeds the anti-bot. Remaining backlog is picked up next cycle.
    # Backlog auto-apply runs only when auto-apply is enabled. Frozen by
    # default (AUTO_APPLY_ENABLED) — when off, nothing is applied automatically,
    # everything waits for manual review.
    if not AUTO_APPLY_ENABLED:
        return
    candidates = get_auto_apply_candidates()
    applied_ids = {v["id"] for v in _stats["auto_applied"]}
    backlog_done = 0
    for v in candidates:
        if v["id"] in applied_ids:
            continue
        if backlog_done >= MAX_VACANCIES_PER_CYCLE:
            logger.info(
                "Autopilot: backlog auto-apply cap (%d) reached this cycle, "
                "%d candidate(s) deferred to next cycle",
                MAX_VACANCIES_PER_CYCLE, len(candidates) - backlog_done,
            )
            break
        try:
            success = await auto_apply(hh_client, v)
            if not success:
                _stats["errors"] += 1
        except Exception as e:
            logger.error("Autopilot auto-apply error for %s: %s", v["id"], e)
            _stats["errors"] += 1
        backlog_done += 1
        await asyncio.sleep(VACANCY_PAUSE)


def _is_blacklisted(vacancy: dict) -> tuple[bool, str | None]:
    """Defensive guard: re-check company/title blacklist at apply time.

    Why duplicated from run_search_cycle: vacancies can arrive at
    auto_apply via TWO paths — the live search loop (where the checks
    above run) AND `get_auto_apply_candidates` (legacy DB rows scored
    under an older, weaker blacklist). The second path bypassed those
    checks entirely; this guard closes that hole.
    """
    company_lower = (vacancy.get("company") or "").lower()
    if any(bl in company_lower for bl in COMPANY_BLACKLIST):
        return True, f"company blacklist: {vacancy.get('company')}"

    title_lower = (vacancy.get("title") or "").lower()
    matched = next((bl for bl in TITLE_BLACKLIST if bl in title_lower), None)
    if matched:
        return True, f"title blacklist match: '{matched}' in '{vacancy.get('title')}'"

    return False, None


async def auto_apply(hh_client: HHClient, vacancy: dict) -> bool:
    """Generate cover letter and apply to vacancy."""
    # Defensive: re-check blacklist even for vacancies pulled from DB
    # via get_auto_apply_candidates (which doesn't know about blacklists).
    blocked, reason = _is_blacklisted(vacancy)
    if blocked:
        logger.warning(
            "Auto-apply BLOCKED for %s (%s) — %s",
            vacancy.get("id"), vacancy.get("title"), reason,
        )
        mark_skipped(vacancy["id"])
        return False

    try:
        letter = await generate_cover_letter(vacancy)
        if not letter:
            logger.warning("Auto-apply: empty cover letter for %s", vacancy["id"])
            return False

        async with HH_LOCK:
            success = await hh_client.apply_to_vacancy(vacancy["id"], letter)
        if success:
            save_response(vacancy["id"], letter, "sent")
            _stats["auto_applied"].append({
                "id": vacancy["id"],
                "title": vacancy.get("title", "?"),
                "company": vacancy.get("company", "?"),
                "score": vacancy.get("relevance_score", 0),
                "url": vacancy.get("url", ""),
            })
            log_action("auto_applied", f"vacancy={vacancy['id']}")
            return True
        else:
            return False
    except Exception as e:
        logger.error("Auto-apply error for %s: %s", vacancy["id"], e)
        return False


async def send_summary(bot: Bot):
    """Send accumulated summary to Telegram."""
    applied = _stats["auto_applied"]

    if not _stats["found"] and not applied:
        # nothing happened, don't spam
        _reset_stats()
        return

    applied_text = ""
    if applied:
        for v in applied:
            applied_text += f"\n{v['title']} @ {v['company']} ({v['score']}/100)\n{v['url']}"
    else:
        applied_text = "\nнет"

    now = datetime.now().strftime("%H:%M %d.%m")
    text = (
        f"Автопилот [{now}]\n\n"
        f"Новых вакансий: {_stats['found']}\n"
        f"Автооткликов (90+): {len(applied)}{applied_text}\n\n"
        f"На ручной просмотр (10-89 + overlap-rescue): {_stats['manual_review']}\n"
        f"Автопропуск (<10): {_stats['auto_skipped']}\n"
        f"Ошибки: {_stats['errors']}"
    )

    try:
        await bot.send_message(TELEGRAM_ADMIN_ID, text)
    except Exception as e:
        logger.error("Summary error: %s", e)

    _reset_stats()
    log_action("autopilot_summary", f"applied={len(applied)}")
