from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfStreamError, PdfReadError
from docx import Document


CVS_DIR = Path("data/samples/cvs")
OUT_DIR = Path("data/outputs/extracted_text")


def safe_console(text: str) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def extract_from_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path), strict=False)
    except Exception:
        with open(file_path, "rb") as f:
            reader = PdfReader(f, strict=False)

    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def extract_from_docx(file_path: Path) -> str:
    doc = Document(str(file_path))
    parts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(parts).strip()


def process_files() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    files = sorted([p for p in CVS_DIR.glob("*") if p.suffix.lower() in [".pdf", ".docx"]])

    for file in files:
        print(f"Processing: {safe_console(file.name)}")
        try:
            if file.suffix.lower() == ".pdf":
                text = extract_from_pdf(file)
            else:
                text = extract_from_docx(file)

            if not text.strip():
                raise ValueError("Empty text extracted")

            out_path = OUT_DIR / f"{file.stem}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"Extracted: {safe_console(out_path.name)}")

        except (PdfStreamError, PdfReadError, ValueError) as e:
            failed.append(f"{file.name} | {type(e).__name__}: {e}")
            print(f"Failed: {safe_console(file.name)} | {type(e).__name__}: {e}")
        except Exception as e:
            failed.append(f"{file.name} | {type(e).__name__}: {e}")
            print(f"Failed: {safe_console(file.name)} | {type(e).__name__}: {e}")

    if failed:
        (OUT_DIR / "_failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"Failed files: {len(failed)} (see {safe_console(str(OUT_DIR / '_failed.txt'))})")


if __name__ == "__main__":
    process_files()