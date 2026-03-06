from __future__ import annotations

import os
import sys
import types
import logging
from pathlib import Path
from datetime import datetime
import unicodedata
import re
import hashlib

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
import pandas as pd


JD_FILE = Path("data/samples/jd/job.txt")
CHUNKS_DIR = Path("data/outputs/chunks")
PERSIST_DIR = "data/vectorstore"
COLLECTION_NAME = "resume_chunks"

EMBED_DIM = 384
TOP_N = 2

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


def safe_console(text: str) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


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


def main() -> None:
    jd_text = JD_FILE.read_text(encoding="utf-8", errors="replace").strip()
    q_emb = hash_embed(jd_text)

    client = chromadb.Client(
        Settings(
            persist_directory=PERSIST_DIR,
            is_persistent=True,
            anonymized_telemetry=False,
        )
    )

    collection = client.get_collection(COLLECTION_NAME)

    rows = []

    for file in CHUNKS_DIR.glob("*_chunks.txt"):
        source = file.stem

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=min(10, collection.count()),
            where={"source": source},
            include=["distances"],
        )

        distances = results["distances"][0] if results.get("distances") else []
        if not distances:
            continue

        best = sorted(distances)[:TOP_N]
        avg_distance = sum(best) / len(best)
        score = 1 / (1 + avg_distance)

        rows.append(
            {
                "cv_source": source,
                "avg_distance_topN": round(avg_distance, 4),
                "score": round(score, 4),
            }
        )

    df = pd.DataFrame(rows).sort_values(by="score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    out_dir = Path("data/outputs/ranking")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ranking_results.csv"
    json_path = out_dir / "ranking_results.json"

    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"ranking_results_{ts}.csv"
        json_path = out_dir / f"ranking_results_{ts}.json"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient="records", indent=2, force_ascii=False)
        print(safe_console(f"ranking_results.csv locked, wrote: {csv_path.name}"))

    try:
        print(df.to_string(index=False))
    except UnicodeEncodeError:
        print(safe_console(df.to_string(index=False)))


if __name__ == "__main__":
    main()