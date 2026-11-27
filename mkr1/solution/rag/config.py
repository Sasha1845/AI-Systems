"""Налаштування рішення. Значення можна перевизначити у файлі .env (див. .env.example)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv необов'язковий
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent.parent  # папка solution/
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "docs"))
INDEX_PATH = BASE_DIR / ".index" / "chunks.json"
# Журнал реальних запитів одразу пишеться в evidence/ — це доказ використання для звіту
LOG_PATH = Path(os.getenv("LOG_PATH", BASE_DIR.parent / "evidence" / "usage_log.jsonl"))

# Пошук
TOP_K = int(os.getenv("TOP_K", "5"))
# Частка (за вагою) термінів питання, яку має містити фрагмент. Відсікає питання «не по темі».
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.3"))

# Генерація відповіді: "ollama" (локально, безкоштовно) або "claude" (потрібен ANTHROPIC_API_KEY)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))  # за замовчуванням Ollama бере менше й обрізає промпт
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))  # перший запит довгий: модель завантажується

# Claude (необов'язково)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
# Ціна за 1 млн токенів, USD. Дефолт — тариф Claude Haiku 5.5 для промптів до 100K токенів.
PRICE_IN_PER_MTOK = float(os.getenv("PRICE_IN_PER_MTOK", "0.10"))
PRICE_OUT_PER_MTOK = float(os.getenv("PRICE_OUT_PER_MTOK", "0.50"))


def default_model(provider: str) -> str:
    return CLAUDE_MODEL if provider == "claude" else OLLAMA_MODEL
