import re
from pathlib import Path


INPUT_DIR = Path("data/outputs/extracted_text")
OUTPUT_DIR = Path("data/outputs/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start = max(end - overlap, 0)
        if end == n:
            break

    return chunks


def process_all() -> None:
    for file in INPUT_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(text)
        chunks = chunk_text(text)

        out = OUTPUT_DIR / f"{file.stem}_chunks.txt"
        with out.open("w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks, start=1):
                f.write(f"--- CHUNK {i} ---\n{ch}\n\n")

        print(f"Chunked: {file.name} -> {len(chunks)} chunks")


if __name__ == "__main__":
    process_all()