"""Читання документів (PDF, DOCX, MD, TXT) і нарізання на фрагменти (chunks)."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}


@dataclass
class Chunk:
    source: str  # шлях до файлу відносно папки docs
    page: str    # номер сторінки для PDF, "-" для інших форматів
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def _omml(el) -> str:
    """Формула Word (OMML) -> звичайний текст: дроби (a)/(b), індекси a_i, степені a^2, корені √(…)."""
    tag = el.tag

    def sub(name):
        found = el.find(M + name)
        return _omml(found) if found is not None else ""

    def chr_of(pr_name, default):
        pr = el.find(M + pr_name)
        ch = pr.find(M + "chr") if pr is not None else None
        return ch.get(M + "val", default) if ch is not None else default

    if tag == M + "t":
        return el.text or ""
    if tag == M + "f":
        return f"({sub('num')})/({sub('den')})"
    if tag == M + "sSub":
        return f"{sub('e')}_{sub('sub')}"
    if tag == M + "sSup":
        return f"{sub('e')}^{sub('sup')}"
    if tag == M + "sSubSup":
        return f"{sub('e')}_{sub('sub')}^{sub('sup')}"
    if tag == M + "rad":
        deg = sub("deg")
        return f"{deg}√({sub('e')})" if deg else f"√({sub('e')})"
    if tag == M + "d":
        beg, end = "(", ")"
        pr = el.find(M + "dPr")
        if pr is not None:
            b_, e_ = pr.find(M + "begChr"), pr.find(M + "endChr")
            beg = b_.get(M + "val", "") if b_ is not None else "("
            end = e_.get(M + "val", "") if e_ is not None else ")"
        return beg + ", ".join(_omml(c) for c in el.findall(M + "e")) + end
    if tag == M + "nary":
        sym = chr_of("naryPr", "∫")
        lo, hi = sub("sub"), sub("sup")
        limits = (f"_({lo})" if lo else "") + (f"^({hi})" if hi else "")
        return f"{sym}{limits} {sub('e')}"
    if tag == M + "bar":
        return f"{sub('e')}\u0304"  # риска над виразом
    if tag == M + "acc":
        return f"{sub('e')}{chr_of('accPr', chr(0x0302))}"
    if tag == M + "func":
        return f"{sub('fName')} {sub('e')}"
    if tag.endswith("Pr"):  # властивості форматування — не текст
        return ""
    return "".join(_omml(child) for child in el)


def _para_text(p) -> str:
    """Текст абзацу в порядку документа: звичайний текст + формули."""
    out: list[str] = []

    def walk(node):
        for child in node:
            if child.tag == W + "t":
                out.append(child.text or "")
            elif child.tag == W + "tab":
                out.append(" ")
            elif child.tag == M + "oMath":
                out.append(" " + _omml(child).strip() + " ")
            else:
                walk(child)

    walk(p)
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


def _docx_text(document) -> str:
    """Текст DOCX у порядку документа: абзаци, формули й таблиці (рядки через « ; », клітинки через « | »)."""
    from docx.table import Table

    parts: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            parts.append(_para_text(child))
        elif tag == "tbl":
            rows = []
            for row in Table(child, document).rows:
                cells: list[str] = []
                for cell in row.cells:
                    text = " ".join(_para_text(p) for p in cell._tc.iterchildren(W + "p")).strip()
                    if text and (not cells or cells[-1] != text):  # об'єднані клітинки повторюються
                        cells.append(text)
                if cells:
                    rows.append(" | ".join(cells))
            parts.append(" ; ".join(rows))
    return "\n\n".join(parts)


def read_pages(path: Path) -> list[tuple[str, str]]:
    """Повертає список (мітка_сторінки, текст)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(str(i + 1), page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if ext == ".docx":
        import docx

        return [("-", _docx_text(docx.Document(str(path))))]
    if ext in {".md", ".txt"}:
        return [("-", path.read_text(encoding="utf-8", errors="ignore"))]
    return []


def _units(text: str, max_chars: int) -> list[str]:
    """Розбиває текст на абзаци/речення, кожен не довший за max_chars."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text.replace("\r", "")):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        buf = ""
        for sentence in re.split(r"(?<=[.!?…])\s+", para):
            while len(sentence) > max_chars:  # дуже довге «речення» — ріжемо жорстко
                if buf:
                    units.append(buf)
                    buf = ""
                units.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if buf and len(buf) + 1 + len(sentence) > max_chars:
                units.append(buf)
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            units.append(buf)
    return units


def split_text(text: str, max_chars: int = 900, overlap: int = 200) -> list[str]:
    """Збирає абзаци у фрагменти до max_chars; останній короткий абзац переходить у наступний фрагмент."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for unit in _units(text, max_chars):
        if cur and cur_len + len(unit) + 1 > max_chars:
            chunks.append("\n".join(cur))
            tail = cur[-1] if len(cur[-1]) <= overlap else ""
            cur = [tail] if tail else []
            cur_len = len(tail)
        cur.append(unit)
        cur_len += len(unit) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading, body = "", []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if "".join(body).strip():
                sections.append((heading, "\n".join(body)))
            heading, body = line.lstrip("#").strip(), []
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((heading, "\n".join(body)))
    return sections


def load_chunks(docs_dir: Path) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if any(part.startswith(".") for part in path.relative_to(docs_dir).parts):
            continue
        rel = path.relative_to(docs_dir).as_posix()
        try:
            pages = read_pages(path)
        except Exception as exc:  # пошкоджений файл не повинен зупиняти індексацію
            print(f"[!] Пропускаю {rel}: {exc}")
            continue
        for page, text in pages:
            if path.suffix.lower() == ".md":
                for heading, body in _markdown_sections(text):
                    for piece in split_text(body):
                        chunks.append(Chunk(rel, page, f"{heading}\n{piece}" if heading else piece))
            else:
                for piece in split_text(text):
                    chunks.append(Chunk(rel, page, piece))
    return chunks
