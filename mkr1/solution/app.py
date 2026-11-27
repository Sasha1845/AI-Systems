"""Веб-інтерфейс (чат).  Запуск: streamlit run app.py"""
import html

import streamlit as st

from rag import config
from rag.loader import SUPPORTED
from rag.pipeline import Assistant

st.set_page_config(page_title="Помічник по матеріалах курсу", page_icon=":material/menu_book:", layout="centered")


@st.cache_resource
def get_assistant() -> Assistant:
    return Assistant()


assistant = get_assistant()

with st.sidebar:
    st.header("Налаштування")
    k = st.slider("Скільки фрагментів передавати моделі", 1, 10, config.TOP_K)
    provider = st.selectbox("Джерело відповідей", ["ollama", "claude"],
                            index=0 if config.LLM_PROVIDER != "claude" else 1)
    model = st.text_input("Модель", config.default_model(provider), key=f"model_{provider}")
    search_only = st.checkbox("Лише пошук (без моделі, миттєво)", value=False)

    st.divider()
    st.subheader("Матеріали")
    uploads = st.file_uploader(
        "Додати файли", type=[e.lstrip(".") for e in SUPPORTED], accept_multiple_files=True
    )
    if uploads:
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for up in uploads:
            (config.DOCS_DIR / up.name).write_bytes(up.getbuffer())
        st.success(f"Збережено файлів: {len(uploads)}. Натисніть «Переіндексувати».")
    if st.button("Переіндексувати"):
        try:
            stats = assistant.build_index()
            st.success(f"Файлів: {stats['files']}, фрагментів: {stats['chunks']}")
        except RuntimeError as exc:
            st.error(str(exc))

def index_stats() -> tuple[int, int]:
    """Скільки файлів і фрагментів зараз в індексі (для шапки)."""
    try:
        assistant.ensure_index()
    except RuntimeError:
        return 0, 0
    chunks = assistant.index.chunks
    return len({c.source for c in chunks}), len(chunks)


# ---------- оформлення (лише шапка і підказка; вивід чату нижче не змінюється) ----------
def _svg(paths: str) -> str:
    return ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + paths + "</svg>")


ICON_BOOK = _svg('<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 '
                 '4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>')
ICON_LOCK = _svg('<rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>')
ICON_COIN = _svg('<circle cx="12" cy="12" r="10"/><path d="M16 8h-6a2 2 0 1 0 0 4h4a2 2 0 1 1 0 4H8M12 18V6"/>')
ICON_CLIP = _svg('<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 '
                 '8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>')
ICON_SEARCH = _svg('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>')
ICON_CPU = _svg('<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9"/>'
                '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>')
ICON_FILE = _svg('<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 '
                 '0 0 2 2h4"/><path d="M10 9H8M16 13H8M16 17H8"/>')

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 820px;}
.hero {display:flex; flex-wrap:wrap; gap:22px; justify-content:space-between; align-items:center;
  padding:26px 30px; margin-bottom:18px; border-radius:22px;
  background: linear-gradient(135deg, rgba(99,130,255,.22), rgba(168,110,255,.14) 60%, rgba(40,200,170,.10));
  border:1px solid rgba(130,150,255,.30); box-shadow:0 8px 30px rgba(20,30,80,.25);}
.hero-text {flex:1 1 340px; min-width:0;}
.hero h1 {display:flex; align-items:center; gap:12px; margin:0 0 8px; padding:0; font-size:1.85rem; line-height:1.2;}
.hero h1 svg {width:34px; height:34px; flex:none; color:#6f8cff;}
.hero p {margin:0 0 14px; opacity:.78; font-size:.98rem; line-height:1.45;}
.pill {display:inline-flex; align-items:center; gap:7px; margin:0 8px 8px 0; padding:5px 13px;
  border-radius:999px; font-size:.84rem; background:rgba(128,128,128,.14); border:1px solid rgba(128,128,128,.32);}
.stats {display:flex; gap:12px; flex:0 0 auto;}
.stat {min-width:104px; padding:14px 18px; border-radius:16px; text-align:center;
  background:rgba(128,128,128,.12); border:1px solid rgba(128,128,128,.30);}
.stat b {display:block; font-size:1.9rem; line-height:1.1; font-weight:650;}
.stat span {font-size:.78rem; opacity:.65; letter-spacing:.03em; text-transform:uppercase;}
.model-line {margin:-4px 2px 18px; font-size:.85rem; opacity:.7;}
.model-line code {padding:2px 8px; border-radius:6px;}
.steps {display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:14px; margin:6px 0 10px;}
.step {padding:18px 18px 16px; border-radius:18px; background:rgba(128,128,128,.07);
  border:1px solid rgba(128,128,128,.28); transition:transform .15s ease, border-color .15s ease;}
.step:hover {transform:translateY(-2px); border-color:rgba(130,150,255,.55);}
.step .ico {display:inline-flex; align-items:center; justify-content:center; width:36px; height:36px;
  margin-bottom:10px; border-radius:11px; color:#6f8cff; background:rgba(99,130,255,.16);}
.step h4 {margin:0 0 4px; padding:0; font-size:1rem;}
.step p {margin:0; font-size:.87rem; opacity:.72; line-height:1.45;}
.hint {margin:14px 2px 4px; font-size:.88rem; opacity:.6;}
</style>
"""


def _html(text: str) -> str:
    """Склеює HTML в один рядок, щоб Markdown не сприйняв відступи за блок коду."""
    return " ".join(line.strip() for line in text.splitlines())


files_n, chunks_n = index_stats()
shown_model = html.escape(model)
st.markdown(_html(CSS), unsafe_allow_html=True)
st.markdown(_html(f"""
<div class="hero">
  <div class="hero-text">
    <h1>{ICON_BOOK}<span>Помічник по матеріалах курсу</span></h1>
    <p>Ставте питання українською: відповідь береться лише з ваших конспектів і лабораторних,
    а кожне твердження має посилання на джерело.</p>
    <span class="pill">{ICON_LOCK} працює локально</span>
    <span class="pill">{ICON_COIN} $0 за запит</span>
    <span class="pill">{ICON_CLIP} відповідь із джерелами</span>
  </div>
  <div class="stats">
    <div class="stat"><b>{files_n}</b><span>файлів</span></div>
    <div class="stat"><b>{chunks_n}</b><span>фрагментів</span></div>
  </div>
</div>
"""), unsafe_allow_html=True)
st.markdown(
    f'<div class="model-line">Модель: <code>{shown_model}</code>'
    + (" · режим: лише пошук" if search_only else "") + "</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


question = st.chat_input("Ваше питання по матеріалах…")

if not st.session_state.messages and not question:
    st.markdown(_html(f"""
    <div class="steps">
      <div class="step"><div class="ico">{ICON_SEARCH}</div><h4>1. Пошук</h4>
        <p>BM25 по триграмах знаходить у ваших файлах {k} найбільш підхожих фрагментів.</p></div>
      <div class="step"><div class="ico">{ICON_CPU}</div><h4>2. Відповідь</h4>
        <p>Локальна модель {shown_model} формулює відповідь лише на основі цих фрагментів.</p></div>
      <div class="step"><div class="ico">{ICON_FILE}</div><h4>3. Джерела</h4>
        <p>Кожне твердження має номер [n]. Звіряйте з джерелом і оцінюйте відповідь кнопками 👍 / 👎.</p></div>
    </div>
    <div class="hint">Напишіть питання внизу сторінки.</div>
    """), unsafe_allow_html=True)


def render(msg: dict, idx: int) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return
        if msg.get("warning"):
            st.warning(msg["warning"])
        with st.expander(f"Джерела ({len(msg['sources'])})"):
            for i, s in enumerate(msg["sources"], 1):
                st.markdown(f"**[{i}] {s['source']}**, стор. {s['page']} · score {s['score']}")
                st.text(s["text"])
        st.caption(
            f"{msg['latency_s']} c · режим: {msg['mode']} · токени: "
            f"{msg['in_tokens']}+{msg['out_tokens']} · ${msg['cost_usd']:.5f}"
        )
        col1, col2, _ = st.columns([1, 1, 6])
        if col1.button("👍", key=f"up{idx}"):
            assistant.log_feedback(msg["question"], "correct")
            st.toast("Записано: відповідь правильна")
        if col2.button("👎", key=f"down{idx}"):
            assistant.log_feedback(msg["question"], "wrong")
            st.toast("Записано: відповідь неправильна")


for i, m in enumerate(st.session_state.messages):
    render(m, i)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render(st.session_state.messages[-1], len(st.session_state.messages) - 1)
    try:
        with st.spinner("Шукаю у ваших матеріалах…"):
            res = assistant.ask(question, k=k, use_llm=not search_only, model=model, provider=provider)
        reply = {
            "role": "assistant", "content": res["answer"], "question": question,
            **{key: res[key] for key in
               ("sources", "warning", "latency_s", "mode", "in_tokens", "out_tokens", "cost_usd")},
        }
    except RuntimeError as exc:
        reply = {"role": "assistant", "content": f"⚠️ {exc}", "question": question, "sources": [],
                 "warning": "", "latency_s": 0, "mode": "error", "in_tokens": 0,
                 "out_tokens": 0, "cost_usd": 0.0}
    st.session_state.messages.append(reply)
    render(reply, len(st.session_state.messages) - 1)
