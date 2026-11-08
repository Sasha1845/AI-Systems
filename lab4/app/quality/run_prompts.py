"""Прогоняє тестові промпти через локальну модель і зберігає відповіді в markdown.

Запуск (з теки lab4):
    python app/quality/run_prompts.py --model qwen3:4b --label before
    python app/quality/run_prompts.py --model qwen3:4b --label after

Потрібно: Ollama з завантаженою моделлю, pip install openai.
Результат: app/quality/quality_<label>.md
"""
import argparse
import time
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser(description="Тест якості на 10 промптах")
    p.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="before або after")
    p.add_argument("--prompts", default=str(HERE / "prompts.txt"))
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--reasoning-effort", default=None, choices=["none", "low", "medium", "high"],
                   help="передавайте лише для моделей, що вміють міркувати")
    args = p.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="local", timeout=120)
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    prompts = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]

    lines = [f"# Якість відповідей: {args.label}", "", f"Модель: `{args.model}`, temperature=0, max_tokens={args.max_tokens}", ""]
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] ...", flush=True)
        start = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=0,
                **extra,
            )
            answer = r.choices[0].message.content or "(порожня відповідь)"
            info = f"{time.perf_counter() - start:.1f} с, {r.usage.completion_tokens} токенів"
        except Exception as e:
            answer, info = f"ПОМИЛКА: {e}", "-"
        lines += [f"## Промпт {i}", "", f"> {prompt}", "", f"Відповідь ({info}):", "", answer.strip(), ""]

    out = HERE / f"quality_{args.label}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Збережено: {out}")


if __name__ == "__main__":
    main()
