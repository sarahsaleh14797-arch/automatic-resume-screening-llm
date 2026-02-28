from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# عدّلي هذا المسار حسب مكان ملف الجوب عندك
JD_FILE = Path("data/samples/jd/job.txt")

TOP_K = 5
PERSIST_DIR = "data/vectorstore"
COLLECTION_NAME = "resume_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    if not JD_FILE.exists():
        raise FileNotFoundError(f"Job description not found: {JD_FILE}")

    jd_text = JD_FILE.read_text(encoding="utf-8", errors="ignore").strip()
    if not jd_text:
        raise ValueError("Job description file is empty.")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = model.encode(jd_text).tolist()

    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, is_persistent=True))
    col = client.get_collection(COLLECTION_NAME)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print("\n=== TOP MATCHED CHUNKS ===")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        print(f"\n#{i} | source={meta.get('source')} | distance={dist:.4f}")
        print(doc[:400])


if __name__ == "__main__":
    main()