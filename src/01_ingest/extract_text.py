from pathlib import Path

from docx import Document
from pypdf import PdfReader


INPUT_DIR = Path("data/samples/cvs")
OUTPUT_DIR = Path("data/outputs/extracted_text")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_from_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    parts = []
    for page in reader.pages:
        parts.append((page.extract_text() or "") + "\n")
    return "".join(parts)


def extract_from_docx(file_path: Path) -> str:
    doc = Document(str(file_path))
    return "\n".join([p.text for p in doc.paragraphs])


def process_files() -> None:
    for file in INPUT_DIR.glob("*"):
        suffix = file.suffix.lower()
        if suffix == ".pdf":
            text = extract_from_pdf(file)
        elif suffix == ".docx":
            text = extract_from_docx(file)
        else:
            continue

        out = OUTPUT_DIR / f"{file.stem}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"Extracted: {file.name}")


if __name__ == "__main__":
    process_files()