"""Генерація відповіді лише на основі знайдених фрагментів: Ollama (локально) або Claude API."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from . import config

SYSTEM_PROMPT = (
    "Ти — помічник для підготовки до навчання. Відповідай ЛИШЕ на основі наданих фрагментів "
    "матеріалів курсу. Відповідай мовою питання повними реченнями, стисло і по суті. "
    "Номер фрагмента став у квадратних дужках ПІСЛЯ твердження, наприклад «Ентропія максимальна, "
    "коли символи рівноймовірні [1].». НІКОЛИ не відповідай самими номерами фрагментів. "
    "Якщо у фрагментах немає відповіді, напиши: «У матеріалах цього немає» — і не вигадуй нічого від себе."
)

RETRY_NOTE = "Відповідь має бути повним реченням зі словами, а не лише номерами фрагментів."
NO_ANSWER = "Модель не змогла сформулювати відповідь. Дивіться знайдені фрагменти нижче."


def build_user_prompt(question: str, hits: list, extra: str = "") -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[{i}] (файл: {hit['source']}, стор.: {hit['page']})\n{hit['text']}")
    fragments = "\n\n".join(parts)
    note = f"Є рівно {len(hits)} фрагментів, пронумерованих [1]–[{len(hits)}]. Посилайся лише на ці номери."
    extra = f"\n{extra}" if extra else ""
    return f"<fragments>\n{fragments}\n</fragments>\n\n{note}{extra}\n\nПитання: {question}"


def strip_bad_citations(text: str, n: int) -> str:
    """Прибирає посилання [k] на неіснуючі фрагменти (мала модель інколи плутає їх з нумерацією в документі)."""
    return re.sub(r"\s?\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", text)


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask_ollama(question: str, hits: list, model: str | None = None, extra: str = "") -> dict:
    model = model or config.OLLAMA_MODEL
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, hits, extra)},
        ],
        "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX, "num_predict": config.MAX_TOKENS},
    }
    try:
        data = _post_json(f"{config.OLLAMA_URL}/api/chat", payload, config.OLLAMA_TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(f"Модель «{model}» не знайдена в Ollama. Виконайте: ollama pull {model}") from exc
        raise
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Не вдалося підключитись до Ollama ({config.OLLAMA_URL}). Запустіть Ollama і виконайте: ollama pull {model}"
        ) from exc
    text = data["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # моделі з «міркуванням»
    return {
        "answer": text,
        "in_tokens": int(data.get("prompt_eval_count") or 0),
        "out_tokens": int(data.get("eval_count") or 0),
    }


def ask_claude(question: str, hits: list, model: str | None = None, extra: str = "") -> dict:
    import anthropic  # потрібен лише для LLM_PROVIDER=claude

    client = anthropic.Anthropic(max_retries=2, timeout=60.0)  # ключ береться з ANTHROPIC_API_KEY
    response = client.messages.create(
        model=model or config.CLAUDE_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(question, hits, extra)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return {
        "answer": text,
        "in_tokens": response.usage.input_tokens,
        "out_tokens": response.usage.output_tokens,
    }


def provider_ready(provider: str) -> str:
    """Порожній рядок, якщо можна викликати; інакше — причина, чому не можна."""
    if provider == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        return "Немає ANTHROPIC_API_KEY — працюю в режимі «лише пошук»."
    return ""


def only_citations(text: str) -> bool:
    """True, якщо у відповіді немає слів, лише номери фрагментів на кшталт «[2] та [4]»."""
    return not re.search(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]{3,}", re.sub(r"\[\d+\]", "", text))


def generate(question: str, hits: list, provider: str, model: str | None = None) -> dict:
    ask = ask_claude if provider == "claude" else ask_ollama
    reply = ask(question, hits, model)
    reply["answer"] = strip_bad_citations(reply["answer"], len(hits))
    if only_citations(reply["answer"]):  # мала модель інколи відповідає лише номерами — пробуємо ще раз
        again = ask(question, hits, model, extra=RETRY_NOTE)
        again["answer"] = strip_bad_citations(again["answer"], len(hits))
        again["in_tokens"] += reply["in_tokens"]
        again["out_tokens"] += reply["out_tokens"]
        reply = again
        if only_citations(reply["answer"]):
            reply["answer"] = NO_ANSWER
    return reply


def cost_usd(provider: str, in_tokens: int, out_tokens: int) -> float:
    if provider != "claude":
        return 0.0  # локальна модель: безкоштовно
    return (in_tokens * config.PRICE_IN_PER_MTOK + out_tokens * config.PRICE_OUT_PER_MTOK) / 1_000_000
