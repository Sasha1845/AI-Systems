"""Головний сценарій: питання -> пошук BM25 -> відповідь Claude -> журнал."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from . import config, llm
from .bm25 import BM25Index
from .loader import load_chunks


class Assistant:
    def __init__(self, docs_dir=None, index_path=None, log_path=None):
        self.docs_dir = Path(docs_dir or config.DOCS_DIR)
        self.index_path = Path(index_path or config.INDEX_PATH)
        self.log_path = Path(log_path or config.LOG_PATH)
        self.index: BM25Index | None = None

    # ---------- індекс ----------
    def build_index(self) -> dict:
        chunks = load_chunks(self.docs_dir)
        if not chunks:
            raise RuntimeError(
                f"У папці {self.docs_dir} немає файлів .pdf/.docx/.md/.txt з текстом. "
                "Покладіть туди свої матеріали."
            )
        self.index = BM25Index(chunks)
        self.index.save(self.index_path)
        return {"files": len({c.source for c in chunks}), "chunks": len(chunks)}

    def ensure_index(self) -> None:
        if self.index is not None:
            return
        if self.index_path.exists():
            self.index = BM25Index.load(self.index_path)
        else:
            self.build_index()

    # ---------- питання ----------
    def ask(self, question: str, k: int | None = None, use_llm: bool = True,
            model: str | None = None, log: bool = True, provider: str | None = None) -> dict:
        started = time.perf_counter()
        provider = (provider or config.LLM_PROVIDER).lower()
        model = model or config.default_model(provider)
        self.ensure_index()
        found = self.index.search(question, k or config.TOP_K, min_coverage=config.MIN_COVERAGE)
        sources = [
            {"source": c.source, "page": c.page, "score": round(s, 3), "text": c.text}
            for c, s in found
        ]
        result = {
            "question": question, "answer": "", "mode": "retrieval-only", "sources": sources,
            "in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0, "warning": "",
            "provider": provider, "model": model,
        }

        not_ready = llm.provider_ready(provider)
        if not sources:
            result["answer"] = "У матеріалах цього немає (пошук не знайшов жодного релевантного фрагмента)."
        elif not use_llm:
            result["answer"] = "Режим «лише пошук»: нижче найрелевантніші фрагменти матеріалів."
        elif not_ready:
            result["warning"] = not_ready
            result["answer"] = "Модель недоступна. Нижче найрелевантніші фрагменти матеріалів."
        else:
            try:
                reply = llm.generate(question, sources, provider, model)
                result.update(
                    answer=reply["answer"], mode="llm",
                    in_tokens=reply["in_tokens"], out_tokens=reply["out_tokens"],
                    cost_usd=llm.cost_usd(provider, reply["in_tokens"], reply["out_tokens"]),
                )
                if provider == "ollama" and reply["in_tokens"] >= config.OLLAMA_NUM_CTX - 8:
                    result["warning"] = (
                        "Промпт заповнив усе вікно контексту — Ollama могла обрізати фрагменти. "
                        "Зменште TOP_K або збільште OLLAMA_NUM_CTX."
                    )
            except Exception as exc:  # fallback: Ollama не запущена, мережа, ліміти, невірний ключ
                result["warning"] = f"Помилка моделі ({type(exc).__name__}): {exc}"
                result["answer"] = "Модель зараз недоступна. Нижче найрелевантніші фрагменти матеріалів."

        result["latency_s"] = round(time.perf_counter() - started, 2)
        if log:
            self._log({"type": "query", **{k_: v for k_, v in result.items()}})
        return result

    # ---------- журнал ----------
    def log_feedback(self, question: str, verdict: str) -> None:
        self._log({"type": "feedback", "question": question, "verdict": verdict})

    def _log(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now().isoformat(timespec="seconds"), **record}
        if record.get("type") == "query":  # у журнал йдуть лише посилання, не повний текст фрагментів
            record["sources"] = [
                {k: v for k, v in s.items() if k != "text"} for s in record["sources"]
            ]
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
