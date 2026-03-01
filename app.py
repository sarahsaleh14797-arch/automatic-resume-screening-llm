import requests
import streamlit as st

from src.llm.llm_client import generate_response
from src.ui.config import (
    APP_NAME,
    LOGO_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TAGLINE,
)
from src.ui.theme import apply_theme


st.set_page_config(page_title=APP_NAME, layout="wide")
apply_theme()

HEADER_HTML = f"""
<div class="sira-header">
  <div class="sira-left">
    <img class="sira-logo" src="app/static/logo" />
  </div>
  <div class="sira-right">
    <div class="sira-title">{APP_NAME}</div>
    <div class="sira-tagline">{TAGLINE}</div>
    <div class="sira-subline">Local LLM via Ollama • RAG Pipeline • Ranking + Explainability</div>
  </div>
</div>
"""

def _inject_logo_as_static_route():
    import base64
    from pathlib import Path

    p = Path(LOGO_PATH)
    if not p.exists():
        return None

    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = p.suffix.lower().replace(".", "")
    mime = "png" if ext == "png" else "svg+xml" if ext == "svg" else "png"
    return f"data:image/{mime};base64,{b64}"


logo_data_url = _inject_logo_as_static_route()

if logo_data_url is None:
    st.error(f"Logo not found at: {LOGO_PATH}")
else:
    st.markdown(
        f"""
        <style>
          .sira-header {{
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 22px 8px 10px 8px;
            margin-top: 6px;
          }}
          .sira-logo {{
            width: 220px;
            height: auto;
            display: block;
          }}
          .sira-title {{
            font-size: 42px;
            line-height: 1.05;
            font-weight: 800;
            color: var(--sira-gold);
            margin: 0;
            padding: 0;
          }}
          .sira-tagline {{
            font-size: 18px;
            line-height: 1.35;
            font-weight: 600;
            color: var(--sira-gold);
            margin-top: 6px;
          }}
          .sira-subline {{
            font-size: 15px;
            line-height: 1.35;
            font-weight: 500;
            color: #ffffff;
            margin-top: 6px;
            opacity: 0.95;
          }}
          @media (max-width: 900px) {{
            .sira-header {{ flex-direction: column; align-items: flex-start; }}
            .sira-logo {{ width: 190px; }}
            .sira-title {{ font-size: 34px; }}
          }}
        </style>
        <div class="sira-header">
          <div class="sira-left">
            <img class="sira-logo" src="{logo_data_url}" />
          </div>
          <div class="sira-right">
            <div class="sira-title">{APP_NAME}</div>
            <div class="sira-tagline">{TAGLINE}</div>
            <div class="sira-subline">Local LLM via Ollama • RAG Pipeline • Ranking + Explainability</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

tabs = st.tabs(["Upload", "Run", "Results", "System Check"])

with tabs[0]:
    st.subheader("Upload CVs and Job Description")
    st.file_uploader("Upload CV files (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    st.text_area("Paste Job Description", height=220)

with tabs[1]:
    st.subheader("Run Screening")
    st.info("Run: Extract → Chunk → Embed/Store → Retrieve → Rank → Explain")

with tabs[2]:
    st.subheader("Results and Explainability")
    st.info("Ranking outputs will appear under data/outputs/ranking")

with tabs[3]:
    st.subheader("System Check")
    ollama_ok = False
    model_ok = False

    try:
        r = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_ok = True
            data = r.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            model_ok = any(OLLAMA_MODEL == m or m.startswith(OLLAMA_MODEL) for m in models)
    except Exception:
        pass

    st.write("### Runtime Status")
    st.write(f"- Ollama service: {'OK' if ollama_ok else 'Not reachable'}")
    st.write(f"- Required model ({OLLAMA_MODEL}): {'Available' if model_ok else 'Missing'}")

    if not ollama_ok:
        st.error("Ollama is not running. Start Ollama and refresh this page.")
    elif not model_ok:
        st.warning(f"Model not found. Run in terminal: ollama pull {OLLAMA_MODEL}")
    else:
        st.success("System prerequisites look good.")

    st.markdown("### Test Local LLM")
    test_prompt = st.text_area("Enter a test prompt:", height=120)

    if st.button("Run LLM Test"):
        if test_prompt.strip() == "":
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating response..."):
                result = generate_response(test_prompt)
            st.write(result)