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


JD_FILE = Path("data/samples/jd/job.txt")
PERSIST_DIR = "data/vectorstore"
COLLECTION_NAME = "resume_chunks"
TOP_K = 5
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


def main() -> None:
    jd_text = JD_FILE.read_text(encoding="utf-8").strip()
    q_emb = hash_embed(jd_text)

    client = chromadb.Client(
        Settings(
            persist_directory=PERSIST_DIR,
            is_persistent=True,
            anonymized_telemetry=False,
        )
    )

    collection = client.get_collection(COLLECTION_NAME)
    effective_k = min(TOP_K, collection.count())

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        print(f"{i}. {meta['source']} | {dist:.4f}")
        print(doc[:400])


if __name__ == "__main__":
    main()