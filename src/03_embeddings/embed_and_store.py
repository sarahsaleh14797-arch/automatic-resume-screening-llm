from __future__ import annotations

import os
import sys
import types
import logging
from pathlib import Path
import hashlib
import re
import unicodedata

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
_SEPARATOR_RE = re.compile(r"\n\s*---\s*\n", re.MULTILINE)

_AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)

_AR_MAP = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
    "ـ": "",
})


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.translate(_AR_MAP)
    t = _AR_DIACRITICS_RE.sub("", t)
    t = t.lower()
    return t


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def hash_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    tokens = tokenize(text)
    v = np.zeros(dim, dtype=np.float32)

    for tok in tokens:
        h = hashlib.md5(tok.encode("utf-8", errors="ignore")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] % 2 == 0) else -1.0
        v[idx] += sign

    norm = float(np.linalg.norm(v))
    if norm > 0:
        v /= norm

    return v.tolist()


def split_chunks(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = _SEPARATOR_RE.split(raw)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    client = chromadb.Client(
        Settings(
            persist_directory=PERSIST_DIR,
            is_persistent=True,
            anonymized_telemetry=False,
        )
    )

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, docs, embs, metas = [], [], [], []
    total = 0

    files = sorted([p for p in CHUNKS_DIR.glob("*_chunks.txt") if p.is_file()], key=lambda p: p.name.lower())

    for f in files:
        source = f.stem
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks = split_chunks(text)

        for i, ch in enumerate(chunks):
            doc_id = f"{source}::{i}"
            ids.append(doc_id)
            docs.append(ch)
            embs.append(hash_embed(ch))
            metas.append({"source": source, "chunk_index": i})
            total += 1

            if len(ids) >= 256:
                collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
                ids, docs, embs, metas = [], [], [], []

    if ids:
        collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    try:
        client.persist()
    except Exception:
        pass

    print(f"Total chunks: {total}")
    try:
        print(f"Collection size: {collection.count()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()