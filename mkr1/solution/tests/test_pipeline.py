
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config, llm  # noqa: E402
from rag.loader import split_text  # noqa: E402
from rag.pipeline import Assistant  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"


class FakeAnthropic:
    """Підміна anthropic.Anthropic."""
    last_kwargs = None

    def __init__(self, **_):
        self.messages = self

    def create(self, **kwargs):
        FakeAnthropic.last_kwargs = kwargs
        block = types.SimpleNamespace(type="text", text="Відповідь [1]")
        usage = types.SimpleNamespace(input_tokens=1000, output_tokens=50)
        return types.SimpleNamespace(content=[block], usage=usage)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.assistant = Assistant(docs_dir=DOCS, index_path=tmp / "i.json", log_path=tmp / "log.jsonl")
        self.assistant.build_index()

    def tearDown(self):
        self.tmp.cleanup()

    # ---- розбиття і пошук ----
    def test_split_text_respects_limit(self):
        for chunk in split_text("Речення номер один. " * 200, max_chars=300):
            self.assertLessEqual(len(chunk), 300 + 200)

    def test_docx_tables_are_indexed(self):
        import docx
        from rag.loader import load_chunks

        with tempfile.TemporaryDirectory() as folder:
            document = docx.Document()
            document.add_paragraph("Звіт про кодування.")
            table = document.add_table(rows=2, cols=2)
            for cell, value in zip(table._cells, ["Символ", "Код", "пробіл", "0110101"]):
                cell.text = value
            document.save(Path(folder) / "lab.docx")
            text = " ".join(c.text for c in load_chunks(Path(folder)))
        self.assertIn("пробіл | 0110101", text)

    def test_retrieval_finds_right_file(self):
        res = self.assistant.ask("Для чого потрібен SAVEPOINT?", use_llm=False)
        self.assertEqual(res["sources"][0]["source"], "sample_transactions.md")

    def test_retrieval_handles_inflection(self):
        res = self.assistant.ask("Що вимагає друга нормальна форма?", use_llm=False)
        self.assertEqual(res["sources"][0]["source"], "sample_db_basics.md")

    def test_off_topic_question_has_no_hits(self):
        for q in ["квантова хромодинаміка бозон", "рецепт борщу з буряком", "хто виграв чемпіонат світу з футболу"]:
            res = self.assistant.ask(q, use_llm=False)
            self.assertEqual(res["sources"], [], q)
            self.assertIn("немає", res["answer"])

    # ---- Ollama ----
    def test_ollama_path_with_fake_server(self):
        reply = {"message": {"content": "<think>міркую</think>Відповідь [1]"},
                 "prompt_eval_count": 800, "eval_count": 40}
        with mock.patch.object(llm, "_post_json", return_value=reply) as post:
            res = self.assistant.ask("Що таке deadlock?", provider="ollama")
        self.assertEqual(res["mode"], "llm")
        self.assertEqual(res["answer"], "Відповідь [1]")
        self.assertEqual((res["in_tokens"], res["out_tokens"], res["cost_usd"]), (800, 40, 0.0))
        url, payload, _ = post.call_args[0]
        self.assertTrue(url.endswith("/api/chat"))
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["num_ctx"], config.OLLAMA_NUM_CTX)
        self.assertIn("sample_transactions.md", payload["messages"][1]["content"])

    def test_fallback_when_ollama_is_down(self):
        with mock.patch.object(config, "OLLAMA_URL", "http://127.0.0.1:9"):
            res = self.assistant.ask("Що таке deadlock?", provider="ollama")
        self.assertEqual(res["mode"], "retrieval-only")
        self.assertIn("Ollama", res["warning"])
        self.assertTrue(res["sources"])

    # ---- Claude (необов'язковий провайдер) ----
    def test_claude_without_key_falls_back(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            res = self.assistant.ask("Що таке deadlock?", provider="claude")
        self.assertEqual(res["mode"], "retrieval-only")
        self.assertTrue(res["warning"])

    def test_claude_path_with_fake_client(self):
        try:
            import anthropic
        except ImportError:
            self.skipTest("anthropic не встановлено (потрібен лише для LLM_PROVIDER=claude)")
        with mock.patch.object(anthropic, "Anthropic", FakeAnthropic), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            res = self.assistant.ask("Що таке deadlock?", provider="claude")
        self.assertEqual(res["mode"], "llm")
        self.assertEqual((res["in_tokens"], res["out_tokens"]), (1000, 50))
        self.assertAlmostEqual(res["cost_usd"], 0.000125, places=6)  # 1000*0.10/1M + 50*0.50/1M
        self.assertIn("<fragments>", FakeAnthropic.last_kwargs["messages"][0]["content"])

    # ---- журнал ----
    def test_log_written(self):
        self.assistant.ask("Що таке deadlock?", use_llm=False)
        record = json.loads(self.assistant.log_path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(record["type"], "query")
        self.assertNotIn("text", record["sources"][0])


if __name__ == "__main__":
    unittest.main()
