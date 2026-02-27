import streamlit as st

def apply_theme():
    st.markdown(
        """
        <style>
            .stApp { background: #0F1C2E; color: #F2F2F2; }
            h1, h2, h3, h4, h5, h6, p, span, div { color: #F2F2F2; }
            .gold { color: #C7A76C; font-weight: 700; }
            .badge {
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                border: 1px solid rgba(255,255,255,0.12);
                background: rgba(255,255,255,0.04);
                margin-right: 8px;
            }
            .stButton>button {
                background-color: #1F4E79;
                color: white;
                border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.10);
                padding: 8px 14px;
            }
            .stButton>button:hover {
                background-color: #C7A76C;
                color: #0B1220;
            }
        </style>
        """,
        unsafe_allow_html=True
    )