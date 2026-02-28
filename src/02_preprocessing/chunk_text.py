from pathlib import Path
import re

INPUT_DIR = Path("data/outputs/extracted_text")
OUTPUT_DIR = Path("data/outputs/chunks")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 800   # characters
CHUNK_OVERLAP = 120


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def process_all():
    for file in INPUT_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(text)
        chunks = chunk_text(text)

        out_file = OUTPUT_DIR / f"{file.stem}_chunks.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks, start=1):
                f.write(f"--- CHUNK {i} ---\n")
                f.write(ch + "\n\n")

        print(f"Chunked: {file.name} -> {len(chunks)} chunks")


if __name__ == "__main__":
    process_all()