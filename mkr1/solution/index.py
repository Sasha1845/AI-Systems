
from rag.pipeline import Assistant

if __name__ == "__main__":
    assistant = Assistant()
    stats = assistant.build_index()
    print(f"Готово: {stats['files']} файл(ів), {stats['chunks']} фрагментів. Папка: {assistant.docs_dir}")
