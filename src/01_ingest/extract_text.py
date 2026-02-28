import os
from pathlib import Path
from pypdf import PdfReader
from docx import Document

INPUT_DIR = Path("data/samples/cvs")
OUTPUT_DIR = Path("data/outputs/extracted_text")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def process_files():
    for file in INPUT_DIR.glob("*"):
        if file.suffix.lower() == ".pdf":
            text = extract_from_pdf(file)
        elif file.suffix.lower() == ".docx":
            text = extract_from_docx(file)
        else:
            continue

        output_file = OUTPUT_DIR / (file.stem + ".txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Extracted: {file.name}")


if __name__ == "__main__":
    process_files()