from __future__ import annotations

import json
import os
import time
from pathlib import Path
from datetime import datetime

from src.llm.llm_client import generate_response


JD_FILE = Path("data/samples/jd/job.txt")
RANKING_JSON = Path("data/outputs/ranking/ranking_results.json")
EXTRACTED_DIR = Path("data/outputs/extracted_text")
OUT_FILE = Path("data/outputs/ranking/llm_explanations.json")

TOP_K = int(os.getenv("LLM_TOP_K", "10"))
TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "180"))
MAX_CV_CHARS = int(os.getenv("LLM_MAX_CV_CHARS", "1800"))
MAX_JD_CHARS = int(os.getenv("LLM_MAX_JD_CHARS", "1200"))
NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "220"))


def safe_console(text: str) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_prompt(jd_text: str, cv_text: str, score: float, cv_name: str) -> str:
    return f"""
You are an expert recruitment assistant.
Task: Compare the candidate CV to the job description and provide an explainable decision.

Job Description:
{jd_text}

Candidate CV ({cv_name}):
{cv_text}

Similarity score (higher is better): {score}

Output format:
1) Summary (3-5 lines)
2) Strengths (bullets)
3) Gaps / Missing skills (bullets)
4) Recommendation: Accept / Consider / Reject
5) Short justification (2-3 lines)

Constraints:
- Use only the provided CV and JD.
- Be concise and factual.
""".strip()


def load_existing() -> dict:
    if not OUT_FILE.exists():
        return {}
    try:
        data = json.loads(OUT_FILE.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, list):
            out = {}
            for item in data:
                key = (item.get("cv_source") or item.get("cv_name") or "").strip()
                if key:
                    out[key] = item
            return out
    except Exception:
        return {}
    return {}


def classify_llm_output(text: str) -> tuple[str, str | None]:
    t = (text or "").strip()
    if t.startswith("Error: Ollama request timed out"):
        return "timeout", t
    if t.startswith("Error connecting to Ollama"):
        return "error", t
    if t.startswith("Error:"):
        return "error", t
    return "ok", None


def main() -> None:
    if not JD_FILE.exists():
        raise FileNotFoundError(f"Job description not found: {JD_FILE}")
    if not RANKING_JSON.exists():
        raise FileNotFoundError(f"Ranking file not found: {RANKING_JSON}")

    jd_text = JD_FILE.read_text(encoding="utf-8", errors="replace").strip()
    if not jd_text:
        raise ValueError("Job description file is empty.")
    jd_text = jd_text[:MAX_JD_CHARS]

    ranking = json.loads(RANKING_JSON.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(ranking, list):
        raise ValueError("ranking_results.json format is not a list.")

    ranking = [r for r in ranking if isinstance(r, dict)]
    ranking = [r for r in ranking if (r.get("cv_source") or "").strip() != ""]
    ranking = sorted(ranking, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    if TOP_K > 0:
        ranking = ranking[:TOP_K]

    existing = load_existing()
    total = len(ranking)

    print(safe_console(f"Explainability: candidates={total}, top_k={TOP_K}, timeout={TIMEOUT_SEC}s"))
    print(safe_console(f"Output: {OUT_FILE}"))

    for i, row in enumerate(ranking, start=1):
        source = str(row.get("cv_source", "")).strip()
        score = float(row.get("score", 0.0))
        cv_name = source.replace("_chunks", "")
        cv_txt_path = EXTRACTED_DIR / f"{cv_name}.txt"

        if source in existing:
            print(safe_console(f"[{i}/{total}] skip (already done): {source}"))
            continue

        if not cv_txt_path.exists():
            print(safe_console(f"[{i}/{total}] missing txt: {cv_txt_path.name}"))
            continue

        cv_text = cv_txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not cv_text:
            print(safe_console(f"[{i}/{total}] empty txt: {cv_txt_path.name}"))
            continue

        cv_text = cv_text[:MAX_CV_CHARS]
        prompt = build_prompt(jd_text, cv_text, score, cv_name)

        print(safe_console(f"[{i}/{total}] LLM: {source} (score={score:.4f})"))
        started = time.time()

        llm_out = generate_response(
            prompt,
            timeout_sec=TIMEOUT_SEC,
            num_predict=NUM_PREDICT,
        )

        status, err = classify_llm_output(llm_out)
        took = time.time() - started

        item = {
            "cv_source": source,
            "cv_name": cv_name,
            "score": score,
            "status": status,
            "error": err,
            "took_sec": round(took, 2),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "llm_analysis": (llm_out or "").strip() if status == "ok" else "",
        }

        existing[source] = item
        atomic_write_json(OUT_FILE, list(existing.values()))
        print(safe_console(f"[{i}/{total}] saved ({status}) in {took:.1f}s"))

    print(safe_console(f"Saved: {OUT_FILE}"))


if __name__ == "__main__":
    main()