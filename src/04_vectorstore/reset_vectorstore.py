import shutil
from pathlib import Path


VS_DIR = Path("data/vectorstore")

if VS_DIR.exists():
    shutil.rmtree(VS_DIR)
    print(f"Deleted vectorstore: {VS_DIR}")
else:
    print("Vectorstore not found (already clean).")

VS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Recreated: {VS_DIR}")