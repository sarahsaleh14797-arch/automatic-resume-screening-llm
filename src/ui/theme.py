import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
          :root{
            --sira-bg: #0b1220;
            --sira-card: #111a2e;
            --sira-border: rgba(255,255,255,0.10);
            --sira-text: #e8eefc;
            --sira-muted: rgba(232,238,252,0.75);
            --sira-gold: #C9A227;
          }

          html, body, [class*="css"] {
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Helvetica Neue", sans-serif;
          }

          .stApp {
            background: var(--sira-bg);
            color: var(--sira-text);
          }

          .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
          }

          h1, h2, h3, h4 {
            color: var(--sira-text);
            margin-top: 0.2rem;
          }

          .stTextArea textarea, .stTextInput input, .stFileUploader, .stSelectbox, .stMultiSelect {
            background: var(--sira-card) !important;
            border: 1px solid var(--sira-border) !important;
            color: var(--sira-text) !important;
          }

          .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
          }

          .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid var(--sira-border);
            color: var(--sira-text);
            border-radius: 12px;
            padding: 10px 14px;
          }

          .stTabs [aria-selected="true"] {
            background: var(--sira-card);
            border-color: rgba(201,162,39,0.55);
          }

          .stDivider {
            border-color: var(--sira-border);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )