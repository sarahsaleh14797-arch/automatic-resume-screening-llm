# Automatic Resume Screening (LLM + RAG) — Local Resume Filtering

An end-to-end local resume screening system using a structured RAG-style pipeline with ranking and LLM explainability via Ollama, delivered through a Streamlit UI.

---

## 1) Features

- Fully local/offline (no external API dependency)
- Upload CVs (PDF/DOCX) and paste Job Description from the UI
- Text extraction from PDF/DOCX
- Chunking (segmentation) of CV text
- Deterministic offline hash embeddings (Arabic + English supported)
- Local vector store using ChromaDB (persistent)
- Candidate ranking using vector similarity
- Explainability via local Ollama LLM:
  - Summary
  - Strengths
  - Gaps / Missing skills
  - Recommendation
- Streamlit UI includes:
  - Save Inputs
  - Save JD as Version (timestamped archive)
  - Clear Outputs (Fresh Run) to prevent old-data mixing
  - Results view + report download (DOCX + HTML)

---

## 2) Tech Stack

- Python 3.11
- Streamlit
- ChromaDB (persistent local)
- Offline deterministic hash embeddings (custom)
- Ollama local LLM
- pandas / numpy
- pypdf + python-docx

---

## 3) Requirements

### 3.1 Python
- Python 3.11 is required.

### 3.2 Ollama
- Ollama must be installed and running locally.
- The required model must be pulled (model name is configured in `src/ui/config.py`).

Check Ollama is running:
```bash
curl http://localhost:11434/api/tags
If curl is not available on Windows, use PowerShell:
Invoke-RestMethod http://localhost:11434/api/tags

## 4) Download / Clone Project
git clone <REPO_URL>
cd automatic-resume-screening-llm

## 5) Setup Environment (Windows)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## 6)Ollama Model Setup
Open:
src/ui/config.py

Verify:
OLLAMA_MODEL

Then pull the model:
ollama pull <MODEL_NAME>   --ex..ollama pull llama3.2:3b

## 7) Run the App
streamlit run app.py

## 8) How to Use (UI Workflow)
8.1 Upload Tab

Upload CV files (PDF/DOCX)

Paste Job Description

Click Save Inputs

Saved locations:

CVs:
data/samples/cvs/

Job Description:
data/samples/jd/job.txt

Buttons:

Save JD as Version

Saves a timestamped copy into:
data/samples/jd/history/

Clear Outputs (Fresh Run)

Recommended whenever you change CVs or JD

Clears:

data/outputs/extracted_text/

data/outputs/chunks/

data/outputs/ranking/

data/vectorstore/

8.2 Run Tab

Click Run Pipeline.

Pipeline steps:
Extract → Chunk → Reset Vectorstore → Embed/Store → Rank → LLM Explain

8.3 Results Tab

Shows ranking table

Expand each candidate to view explainability sections

Download evaluation reports:

DOCX

HTML

## 9) Pipeline Steps (Details)
Step 1: Extract

Script:
src/01_ingest/extract_text.py

Output:
data/outputs/extracted_text/*.txt

Step 2: Chunking

Script:
src/02_preprocessing/chunk_text.py

Output:
data/outputs/chunks/*_chunks.txt

Step 3: Reset Vectorstore

Script:
src/04_vectorstore/reset_vectorstore.py

Output:
data/vectorstore/

Step 4: Embed & Store

Script:
src/03_embeddings/embed_and_store.py

Output:
ChromaDB collection: resume_chunks

Step 5: Ranking

Script:
src/07_ranking/rank_cvs.py

Output:
data/outputs/ranking/ranking_results.csv
data/outputs/ranking/ranking_results.json

Step 6: Explainability (LLM)

Script:
python -m src.llm.explain_with_llm

Output:
data/outputs/ranking/llm_explanations.json

## 10)Outputs

Ranking:

data/outputs/ranking/ranking_results.csv

data/outputs/ranking/ranking_results.json

LLM Explainability:

data/outputs/ranking/llm_explanations.json

## 11) Run Pipeline from Terminal (Optional)
python src/01_ingest/extract_text.py
python src/02_preprocessing/chunk_text.py
python src/04_vectorstore/reset_vectorstore.py
python src/03_embeddings/embed_and_store.py
python src/07_ranking/rank_cvs.py
python -m src.llm.explain_with_llm
streamlit run app.py

## 12) Notes

Arabic + English are supported in ranking via tokenization/normalization.

If a PDF is scanned (image-based), text extraction may fail because OCR is not used.

Use Clear Outputs (Fresh Run) when changing CVs or JD to avoid mixing old results.

## 13) Troubleshooting
Ollama Not Reachable

Check the System Check tab

Start Ollama, then refresh the page

Model Missing

Pull the configured model:

ollama pull <MODEL_NAME>
Old Candidates / Old Results Appearing

Use:
Upload → Clear Outputs (Fresh Run)

PermissionError on ranking_results.csv

The file is likely open in Excel

Close Excel and run again

## 14)Project Structure

app.py — Streamlit UI

src/ — pipeline scripts (extract, chunk, embed, rank, llm)

data/samples/cvs/ — uploaded CV files (local only, ignored by git)

data/samples/jd/job.txt — current job description (local only, ignored by git)

data/samples/jd/history/ — archived JD versions (local only, ignored by git)

data/outputs/ — pipeline outputs (local only, ignored by git)

data/vectorstore/ — local ChromaDB persistence (local only, ignored by git)