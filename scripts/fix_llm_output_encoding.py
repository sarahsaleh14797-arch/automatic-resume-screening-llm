from pathlib import Path

p = Path("data/outputs/ranking/llm_explanations.json")
txt = p.read_text(encoding="utf-8", errors="ignore")

# replace common mojibake bullet with hyphen bullet
txt = txt.replace("ÔÇó", "-")

p.write_text(txt, encoding="utf-8")
print("OK: cleaned bullets in", p)