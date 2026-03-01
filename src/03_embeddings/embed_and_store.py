import hashlib
import os
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


CHUNKS_DIR = Path("data/outputs/chunks")
PERSIST_DIR = "data/vectorstore"
COLLECTION_NAME = "resume_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def parse_chunks(file_text: str) -> list[str]:
    parts = file_text.split("--- CHUNK ")
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        content = part.split("\n", 1)[-1].strip()
        if content:
            out.append(content)
    return out


def stable_id(source: str, chunk: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"||")
    h.update(chunk.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def main() -> None:
    if not CHUNKS_DIR.exists():
        raise FileNotFoundError(f"Chunks directory not found: {CHUNKS_DIR}")

    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.Client(
        Settings(
            persist_directory=PERSIST_DIR,
            is_persistent=True,
        )
    )
    collection = client.get_or_create_collection(COLLECTION_NAME)

    total = 0
    for file in CHUNKS_DIR.glob("*_chunks.txt"):
        content = file.read_text(encoding="utf-8", errors="ignore")
        chunks = parse_chunks(content)
        if not chunks:
            print(f"Skipped (no chunks): {file.name}")
            continue

        source = file.stem
        ids = [stable_id(source, ch) for ch in chunks]
        embeddings = embedder.encode(chunks).tolist()
        metadatas = [{"source": source, "file": file.name} for _ in chunks]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total += len(chunks)
        print(f"Embedded+Stored: {file.name} -> {len(chunks)} chunks")

    try:
        client.persist()
    except Exception:
        pass

    print(f"Done. Total chunks upserted: {total}")
    print(f"Collection '{COLLECTION_NAME}' count: {collection.count()}")


if __name__ == "__main__":
    main()