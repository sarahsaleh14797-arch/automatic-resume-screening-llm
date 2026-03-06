from pathlib import Path
import re

IN_DIR = Path("data/outputs/extracted_text")
OUT_DIR = Path("data/outputs/chunks")

CHUNK_SIZE = 1200
OVERLAP = 150


def safe_console(text: str) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def chunk_text(text: str):
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - OVERLAP)

    return chunks


def process_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [
            p
            for p in IN_DIR.glob("*.txt")
            if p.name != "_failed.txt" and not p.name.startswith("_")
        ],
        key=lambda p: p.name.lower(),
    )

    for file in files:
        text = file.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text)

        out_file = OUT_DIR / f"{file.stem}_chunks.txt"
        out_file.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")

        print(f"Chunked: {safe_console(file.name)} -> {len(chunks)} chunks")


if __name__ == "__main__":
    process_all()