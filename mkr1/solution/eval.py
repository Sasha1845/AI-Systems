"""Вимірювання якості на наборі питань -> цифри для таблиці «після» у report.md.

    python eval.py --questions eval/questions.json            # з моделлю (Ollama або Claude, див. .env)
    python eval.py --questions eval/questions.json --no-llm   # лише пошук (швидко, без моделі)

Формат questions.json (список):
    {"q": "Що таке 3НФ?", "source": "lecture3.pdf", "keywords": ["транзитив"]}
  source   — частина назви файлу, де лежить відповідь, напр. "ЛР2", або список варіантів ["ЛР4", "робота_4"]
             (для метрики hit@k, необов'язково)
  keywords — частини слів (корені!), які мають бути у правильній відповіді (необов'язково)
"""
import argparse
import json
from pathlib import Path

from rag.pipeline import Assistant


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--out", default="eval/results.json")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    assistant = Assistant()
    assistant.build_index()  # свіжий індекс по поточних матеріалах

    rows = []
    for item in questions:
        res = assistant.ask(item["q"], k=args.k, use_llm=not args.no_llm, log=False)
        context = " ".join(s["text"] for s in res["sources"]).lower()
        keywords = [w.lower() for w in item.get("keywords", [])]
        rows.append({
            "q": item["q"],
            "mode": res["mode"],
            "hit_at_k": any(
                name in s["source"]
                for s in res["sources"]
                for name in ([item["source"]] if isinstance(item["source"], str) else item["source"])
            ) if "source" in item else None,
            "context_has_answer": all(w in context for w in keywords) if keywords else None,
            "answer_correct": (all(w in res["answer"].lower() for w in keywords)
                               if keywords and res["mode"] == "llm" else None),
            "latency_s": res["latency_s"],
            "in_tokens": res["in_tokens"],
            "out_tokens": res["out_tokens"],
            "cost_usd": res["cost_usd"],
            "answer": res["answer"],
            "warning": res["warning"],
        })

    n = len(rows)
    summary = {
        "questions": n,
        "k": args.k,
        "llm": not args.no_llm,
        "hit_at_k": mean([r["hit_at_k"] for r in rows]),
        "context_has_answer": mean([r["context_has_answer"] for r in rows]),
        "answer_correct": mean([r["answer_correct"] for r in rows]),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "avg_cost_usd": round(sum(r["cost_usd"] for r in rows) / n, 6),
        "avg_in_tokens": round(sum(r["in_tokens"] for r in rows) / n),
        "avg_out_tokens": round(sum(r["out_tokens"] for r in rows) / n),
        "api_errors": sum(1 for r in rows if r["warning"]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| Метрика | Значення |\n|---|---|")
    print(f"| Питань | {n} |")
    print(f"| Потрібний файл у топ-{args.k} (hit@{args.k}) | {pct(summary['hit_at_k'])} |")
    print(f"| Потрібні слова є у знайдених фрагментах | {pct(summary['context_has_answer'])} |")
    print(f"| Правильних відповідей моделі (за ключовими словами) | {pct(summary['answer_correct'])} |")
    print(f"| Середній час відповіді, с | {summary['avg_latency_s']} |")
    print(f"| Середні токени (вхід+вихід) | {summary['avg_in_tokens']}+{summary['avg_out_tokens']} |")
    print(f"| Вартість одного питання, $ | {summary['avg_cost_usd']} |")
    print(f"| Помилок/фолбеків API | {summary['api_errors']} |")
    print(f"\nДетально: {out}")


if __name__ == "__main__":
    main()
