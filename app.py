import streamlit as st
import requests

from src.ui.theme import apply_theme
from src.ui.config import APP_NAME, TAGLINE, LOGO_PATH, OLLAMA_BASE_URL, OLLAMA_MODEL
from src.llm.llm_client import generate_response
st.set_page_config(page_title=APP_NAME, layout="wide")
apply_theme()
col1, col2 = st.columns([1.5, 5], vertical_alignment="center")

with col1:
    st.image(LOGO_PATH, width=170)

with col2:
    st.markdown(
        """
        <div style='line-height:1.1;'>
            <span style='font-size:42px; font-weight:800; letter-spacing:1px;'>
                SIRA <span style='color:#D4AF37;'>CV</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='margin-top:8px; font-size:18px; color:#D4AF37; font-weight:600;'>
            Automatic Resume Screening using LLM
        </div>
        """,
        unsafe_allow_html=True
    )
st.divider()

tabs = st.tabs(["Upload", "Run", "Results", "System Check"])

with tabs[0]:
    st.subheader("Upload CVs + Job Description")
    st.file_uploader("Upload CV files (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    st.text_area("Paste Job Description", height=220)
    st.info("Next steps: Extraction → Chunking → Embeddings → Vector DB → Retrieval → Ollama Scoring → Ranking → Report")

with tabs[1]:
    st.subheader("Run Screening")
    st.warning("Pipeline will be implemented step-by-step to match the approved plan 100%.")

with tabs[2]:
    st.subheader("Results + Explainability")
    st.warning("Ranking + evidence + explanations will appear here.")

with tabs[3]:
    st.subheader("System Check (Doctor/Committee)")

    ollama_ok = False
    model_ok = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_ok = True
            data = r.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            model_ok = any(OLLAMA_MODEL == m or m.startswith(OLLAMA_MODEL) for m in models)
    except Exception:
        pass

    st.write("### Runtime Status")
    st.write(f"- Ollama service: {'✅ OK' if ollama_ok else '❌ Not reachable'}")
    st.write(f"- Required model ({OLLAMA_MODEL}): {'✅ Available' if model_ok else '❌ Missing'}")

    if not ollama_ok:
        st.error("Ollama is not running. Start Ollama and refresh this page.")
    elif not model_ok:
        st.warning(f"Model not found. Run once in terminal: `ollama pull {OLLAMA_MODEL}`")
    else:
        st.success("System prerequisites look good ✅")
st.markdown("### 🔬 Test Local LLM")

test_prompt = st.text_area("Enter a test prompt:")

if st.button("Run LLM Test"):
    if test_prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            result = generate_response(test_prompt)
        st.success("Response:")
        st.write(result)