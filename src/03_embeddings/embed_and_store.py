from __future__ import annotations

import os
import sys
import types
import hashlib
import re
from pathlib import Path
import logging

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"

logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

dummy = types.ModuleType("posthog")
dummy.capture = lambda *args, **kwargs: None
dummy.identify = lambda *args, **kwargs: None
dummy.flush = lambda *args, **kwargs: None
sys.modules["posthog"] = dummy

import chromadb
from chromadb.config import Settings
import numpy as np


CHUNKS_DIR = Path("data/outputs/chunks")
PERSIST_DIR = "data/vectorstore"
COLLECTION_NAME = "resume_chunks"
EMBED_DIM = 384

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def hash_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    tokens = tokenize(text)
    v = np.zeros(dim, dtype=np.float32)

    for tok in tokens:
        h = hashlib.md5(tok.encode()).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] % 2 == 0) else -1.0
        v[idx] += sign

    norm = float(np.linalg.norm(v))
    if norm > 0:
        v /= norm

    return v.tolist()


def parse_chunks(file_text: str) -> list[str]:
    parts = file_text.split("--- CHUNK ")
    out = []
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
    h.update(source.encode())
    h.update(b"||")
    h.update(chunk.encode())
    return h.hexdigest()


def main() -> None:
    client = chromadb.Client(
        Settings(
            persist_directory=PERSIST_DIR,
            is_persistent=True,
            anonymized_telemetry=False,
        )
    )

    collection = client.get_or_create_collection(COLLECTION_NAME)

    total = 0

    for file in CHUNKS_DIR.glob("*_chunks.txt"):
        content = file.read_text(encoding="utf-8")
        chunks = parse_chunks(content)

        source = file.stem
        ids = [stable_id(source, ch) for ch in chunks]
        embeddings = [hash_embed(ch) for ch in chunks]
        metadatas = [{"source": source} for _ in chunks]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total += len(chunks)

    print("Total chunks:", total)
    print("Collection size:", collection.count())


if __name__ == "__main__":
    main()