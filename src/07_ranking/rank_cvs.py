from pathlib import Path
import re
import pandas as pd

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# --- Config ---
JD_FILE = Path("data/samples/jd/job.txt")   
CHUNKS_DIR = Path("data/outputs/chunks")    
PERSIST_DIR = "data/vectorstore"
COLLECTION_NAME = "resume_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_N_CHUNKS_PER_CV = 2   
N_RESULTS_PER_CV = 10     


def safe_slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    return s.strip("_")


def main():
    if not JD_FILE.exists():
        raise FileNotFoundError(f"Job description not found: {JD_FILE}")

    if not CHUNKS_DIR.exists():
        raise FileNotFoundError(f"Chunks directory not found: {CHUNKS_DIR}")

    # Load JD text
    jd_text = JD_FILE.read_text(encoding="utf-8", errors="ignore").strip()
    if not jd_text:
        raise ValueError("Job description file is empty.")

    # Load embedder
    model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = model.encode(jd_text).tolist()

    # Connect to Chroma
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, is_persistent=True))
    col = client.get_collection(COLLECTION_NAME)

    # Discover CV "sources" from chunk files (metadata source we stored = file.stem)
    # Example chunk file: Alaa_Saleh_CV_chunks.txt -> source: Alaa_Saleh_CV_chunks
    sources = []
    for f in CHUNKS_DIR.glob("*_chunks.txt"):
        sources.append(f.stem)

    if not sources:
        raise RuntimeError("No chunk files found. Run preprocessing first.")

    results_rows = []

    for source in sorted(sources):
        # Query only chunks belonging to this CV via metadata filter
        res = col.query(
            query_embeddings=[q_emb],
            n_results=N_RESULTS_PER_CV,
            where={"source": source},
            include=["documents", "metadatas", "distances"],
        )

        docs = res["documents"][0]
        dists = res["distances"][0]

        if not docs:
            # No chunks found for this CV
            continue

        # Take best TOP_N distances (smaller = better)
        best = list(sorted(zip(dists, docs), key=lambda x: x[0]))[:TOP_N_CHUNKS_PER_CV]
        best_dists = [x[0] for x in best]
        best_docs = [x[1] for x in best]

        avg_dist = sum(best_dists) / len(best_dists)

        # Convert distance to a more intuitive score (higher = better)
        # score in (0,1] 
        score = 1.0 / (1.0 + avg_dist)

        results_rows.append({
            "cv_source": source,
            "avg_distance_topN": round(avg_dist, 4),
            "score": round(score, 4),
            "top_chunk_1_preview": (best_docs[0][:180] if len(best_docs) > 0 else ""),
            "top_chunk_2_preview": (best_docs[1][:180] if len(best_docs) > 1 else ""),
        })

    if not results_rows:
        raise RuntimeError("No ranking results produced. Check collection + metadata.")

    df = pd.DataFrame(results_rows).sort_values(by="score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    # Output folder
    out_dir = Path("data/outputs/ranking")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ranking_results.csv"
    json_path = out_dir / "ranking_results.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print("\n=== CV RANKING RESULTS ===")
    print(df[["rank", "cv_source", "avg_distance_topN", "score"]].to_string(index=False))

    print(f"\nSaved:\n- {csv_path}\n- {json_path}")


if __name__ == "__main__":
    main()