import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0"))

# Manual mode: when on, the bot does NO background work — no autopilot
# search/auto-apply, no negotiations polling. It only drafts a cover letter
# when you paste an hh.ru/vacancy/ link, and you send it yourself on hh.ru.
# The auto-apply "Отправить" button is hidden in this mode (drafting only).
# Toggle via MANUAL_MODE in .env (true/false). Default: false (full autopilot).
MANUAL_MODE = os.getenv("MANUAL_MODE", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

# Auto-apply switch. When False (default), the autopilot still SEARCHES and
# sends every relevant vacancy (incl. 90+) to Telegram for manual review, but
# NEVER applies on its own — you review and hit "Откликнуться" yourself. This
# is different from MANUAL_MODE (which stops background search entirely).
# Disabled by default — surface every relevant vacancy for manual review
# instead of applying hands-off. Set AUTO_APPLY_ENABLED=true for auto-apply.
AUTO_APPLY_ENABLED = os.getenv("AUTO_APPLY_ENABLED", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

# hh.ru
HH_LOGIN = os.getenv("HH_LOGIN", "")  # phone number
HH_COOKIES_PATH = BASE_DIR / "hh_cookies.json"
# Optional proxy for hh.ru (Playwright). For bypassing geo-blocking when running
# outside Russia. Formats:
#   socks5://user:pass@host:port
#   http://user:pass@host:port
#   http://host:port  (no auth)
HH_PROXY = os.getenv("HH_PROXY", "")

# GLM (Zhipu AI)
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_PROXY = os.getenv("GLM_PROXY", "")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Resume
RESUME_FILE = BASE_DIR / os.getenv("RESUME_FILE", "resume.txt")

# === Telegram channel reading ===
# Channels are read via the credential-free public web mirror
# (https://t.me/s/<channel>), NOT via Telethon/MTProto. So NO api_id/
# api_hash and NO login are required — see parser/tg_client.py.
#
# TG_USER_* below are OBSOLETE (left only so any old reference doesn't
# crash). They are no longer used anywhere. Safe to delete from .env.
TG_USER_API_ID = int(os.getenv("TG_USER_API_ID", "0"))
TG_USER_API_HASH = os.getenv("TG_USER_API_HASH", "")
TG_USER_SESSION_FILE = BASE_DIR / os.getenv(
    "TG_USER_SESSION_FILE", "data/tg_user.session"
)
TG_CHANNELS_CONFIG = BASE_DIR / "data" / "tg_channels.yaml"

# Optional HTTP/SOCKS proxy for reading t.me (only if t.me is blocked on
# your connection). Format: http://user:pass@host:port. Empty = direct.
TG_HTTP_PROXY = os.getenv("TG_HTTP_PROXY", "")

# === Telegram channel monitoring (the tg_loop background task) ===
# The monitor pulls job channels, extracts + scores posts, and pushes
# relevant ones to your Telegram. It NEVER auto-applies (on Telegram you
# apply by messaging a recruiter yourself) — so it runs even in MANUAL_MODE.
# Channels are read via the credential-free t.me/s/ web mirror (no api keys);
# gated only on TG_MONITOR_ENABLED below.
TG_MONITOR_ENABLED = os.getenv("TG_MONITOR_ENABLED", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
TG_PULL_INTERVAL = int(os.getenv("TG_PULL_INTERVAL", str(30 * 60)))  # 30 min
# How far back each pull looks. Wider than the interval so nothing is missed
# between cycles; tg_seen dedup prevents re-notifying the overlap.
TG_LOOKBACK_HOURS = int(os.getenv("TG_LOOKBACK_HOURS", "6"))
TG_LIMIT_PER_CHANNEL = int(os.getenv("TG_LIMIT_PER_CHANNEL", "30"))

# DB
DB_PATH = BASE_DIR / "data" / "hh_bot.db"

# === Candidate personalisation ===
# CANDIDATE_NAME is the first name used in the cover letter signature
# ("С уважением, <CANDIDATE_NAME>"). The rest of the candidate context
# (resume summary, products, metrics, NDA rules, opening samples) lives
# in `prompts/candidate.txt` (gitignored, your real data) — see
# `prompts/candidate.example.txt` for the template.
CANDIDATE_NAME = os.getenv("CANDIDATE_NAME", "")
