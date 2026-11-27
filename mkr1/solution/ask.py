
import argparse

from rag.pipeline import Assistant


def show(result: dict) -> None:
    print("\n" + result["answer"])
    if result["warning"]:
        print(f"\n[!] {result['warning']}")
    print("\nДжерела:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['source']}, стор. {s['page']} (score {s['score']})")
        if result["mode"] == "retrieval-only":
            print("      " + s["text"][:300].replace("\n", " ") + "…")
    print(
        f"\n{result['latency_s']} c | режим: {result['mode']} | "
        f"токени: {result['in_tokens']}+{result['out_tokens']} | ${result['cost_usd']:.5f}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("-k", type=int, default=None, help="скільки фрагментів брати")
    parser.add_argument("--no-llm", action="store_true", help="лише пошук, без Claude")
    args = parser.parse_args()

    assistant = Assistant()
    if args.question:
        show(assistant.ask(args.question, k=args.k, use_llm=not args.no_llm))
        return
    print("Питайте про свої матеріали (порожній рядок — вихід).")
    while True:
        question = input("> ").strip()
        if not question:
            break
        show(assistant.ask(question, k=args.k, use_llm=not args.no_llm))


if __name__ == "__main__":
    main()
