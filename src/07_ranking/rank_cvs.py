from pathlib import Path
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


JD_FILE = Path("data/samples/jd/job.txt")
CHUNKS_DIR = Path("data/outputs/chunks")

PERSIST_DIR = "data/vectorstore"
COLLECTION_NAME = "resume_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_N_CHUNKS_PER_CV = 2
N_RESULTS_PER_CV = 10


def main() -> None:
    if not JD_FILE.exists():
        raise FileNotFoundError(f"Job description not found: {JD_FILE}")
    if not CHUNKS_DIR.exists():
        raise FileNotFoundError(f"Chunks directory not found: {CHUNKS_DIR}")

    jd_text = JD_FILE.read_text(encoding="utf-8", errors="ignore").strip()
    if not jd_text:
        raise ValueError("Job description file is empty.")

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = embedder.encode(jd_text).tolist()

    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, is_persistent=True))
    col = client.get_collection(COLLECTION_NAME)

    sources = [f.stem for f in CHUNKS_DIR.glob("*_chunks.txt")]
    if not sources:
        raise RuntimeError("No chunk files found. Run preprocessing first.")

    rows = []
    for source in sorted(sources):
        res = col.query(
            query_embeddings=[q_emb],
            n_results=N_RESULTS_PER_CV,
            where={"source": source},
            include=["documents", "distances"],
        )

        docs = res["documents"][0]
        dists = res["distances"][0]
        if not docs:
            continue

        best = sorted(zip(dists, docs), key=lambda x: x[0])[:TOP_N_CHUNKS_PER_CV]
        best_dists = [x[0] for x in best]
        best_docs = [x[1] for x in best]

        avg_dist = sum(best_dists) / len(best_dists)
        score = 1.0 / (1.0 + avg_dist)

        rows.append(
            {
                "cv_source": source,
                "avg_distance_topN": round(avg_dist, 4),
                "score": round(score, 4),
                "top_chunk_1_preview": best_docs[0][:200] if len(best_docs) > 0 else "",
                "top_chunk_2_preview": best_docs[1][:200] if len(best_docs) > 1 else "",
            }
        )

    if not rows:
        raise RuntimeError("No ranking results produced. Check collection and metadata.")

    df = (
        pd.DataFrame(rows)
        .sort_values(by="score", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", range(1, len(df) + 1))

    out_dir = Path("data/outputs/ranking")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ranking_results.csv"
    json_path = out_dir / "ranking_results.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print(df[["rank", "cv_source", "avg_distance_topN", "score"]].to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()