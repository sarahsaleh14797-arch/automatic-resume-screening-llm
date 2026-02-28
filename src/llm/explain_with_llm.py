from __future__ import annotations

import json
from pathlib import Path

from src.llm.llm_client import generate_response

JD_FILE = Path("data/samples/jd/job.txt")
RANKING_JSON = Path("data/outputs/ranking/ranking_results.json")
EXTRACTED_DIR = Path("data/outputs/extracted_text")
OUT_FILE = Path("data/outputs/ranking/llm_explanations.json")


def build_prompt(jd_text: str, cv_text: str, score: float, cv_name: str) -> str:
    return f"""
You are an expert recruitment assistant.

TASK:
Compare the candidate CV to the job description and provide an explainable decision.

JOB DESCRIPTION:
{jd_text}

CANDIDATE CV ({cv_name}):
{cv_text}

Similarity score (higher is better): {score}

OUTPUT FORMAT (strict):
1) Summary (3-5 lines)
2) Strengths (bullet points)
3) Gaps / Missing skills (bullet points)
4) Recommendation: Accept / Consider / Reject
5) Short justification (2-3 lines)

Be concise, factual, and grounded in the provided CV/JD only.
""".strip()


def main():
    if not JD_FILE.exists():
        raise FileNotFoundError(f"Job description not found: {JD_FILE}")

    if not RANKING_JSON.exists():
        raise FileNotFoundError(f"Ranking file not found: {RANKING_JSON}")

    jd_text = JD_FILE.read_text(encoding="utf-8", errors="ignore").strip()
    if not jd_text:
        raise ValueError("Job description file is empty.")

    ranking = json.loads(RANKING_JSON.read_text(encoding="utf-8"))
    explanations = []

    for row in ranking:
        source = row.get("cv_source", "")
        score = float(row.get("score", 0.0))

        cv_name = source.replace("_chunks", "")  # e.g. Alaa_Saleh_CV
        cv_txt_path = EXTRACTED_DIR / f"{cv_name}.txt"

        if not cv_txt_path.exists():
            print(f"Skipped (missing extracted CV text): {cv_txt_path}")
            continue

        cv_text = cv_txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not cv_text:
            print(f"Skipped (empty CV text): {cv_txt_path}")
            continue

        prompt = build_prompt(jd_text, cv_text, score, cv_name)
        print(f"\nGenerating LLM explanation for: {cv_name} ...")

        llm_out = generate_response(prompt)

        explanations.append({
            "cv_source": source,
            "cv_name": cv_name,
            "score": score,
            "llm_analysis": llm_out.strip(),
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(explanations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()