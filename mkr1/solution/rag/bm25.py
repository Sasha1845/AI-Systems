"""Локальний пошук BM25 без зовнішніх залежностей.

Слова розбиваються на символьні триграми: це дає стійкість до відмінків і закінчень
в українській мові («база», «бази», «базами» мають спільні триграми) без стемера і без ембедингів.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .loader import Chunk

WORD = re.compile(r"\w+", re.UNICODE)
STOP = {
    "що", "як", "це", "та", "чи", "які", "який", "яка", "яке", "для", "при", "від", "про",
    "між", "або", "але", "ще", "так", "цей", "ця", "ці", "його", "їх", "має", "мають", "яких",
    "the", "and", "for", "are", "what", "how", "is", "of", "to", "in", "a",
}


def tokenize(text: str) -> list[str]:
    terms: list[str] = []
    for word in WORD.findall(text.lower()):
        if word in STOP or len(word) < 2:
            continue
        if len(word) <= 3:
            terms.append(word)
        else:
            terms.extend(word[i : i + 3] for i in range(len(word) - 2))
    return terms


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            self.doc_len.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings[term].append((i, freq))
        n = len(chunks)
        self.avg_len = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {
            term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self.postings.items()
        }

    def search(self, query: str, k: int = 5, min_coverage: float = 0.0) -> list[tuple[Chunk, float]]:
        """min_coverage — яку частку (за вагою idf) термінів питання має містити фрагмент.
        Відсікає питання «не по темі», які збігаються з документами лише випадковими триграмами."""
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, float] = defaultdict(float)
        terms = set(tokenize(query))
        max_idf = max(self.idf.values(), default=1.0)
        total = sum(self.idf.get(t, max_idf) for t in terms)  # невідомі терміни «коштують» максимум
        for term in terms:
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i, freq in self.postings[term]:
                norm = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * freq * (self.k1 + 1) / norm
                matched[i] += idf
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [
            (self.chunks[i], score)
            for i, score in ranked
            if score > 0 and total > 0 and matched[i] / total >= min_coverage
        ][:k]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Chunk(**d) for d in data])
