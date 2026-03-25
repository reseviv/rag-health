import unicodedata

from pypdf import PdfReader
import re
from pathlib import Path
import json

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/chunks.json")

# read the text from a file
def read_text_file(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text

#split the pdf by page
def extract_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        text = normalise_text(text)
        text = fix_pdf_artifacts(text)
        
        if not text:
            continue

        # keep line breaks, only clean repeated spaces
        text = re.sub(r"[ \t]+", " ", text).strip()

        pages.append({
            "page": i + 1,
            "text": text
        })
    return pages

# text normalization
def normalise_text(text):
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u00a0": " ",
        "\u00ad": "",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def fix_pdf_artifacts(text):
    # join spaced letters like "H o m e" -> "Home"
    text = re.sub(
        r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b",
        lambda m: m.group(0).replace(" ", ""),
        text
    )

    # remove isolated page numbers / roman numerals
    text = re.sub(r"(?m)^\s*(?:\d+|[ivxlcdmIVXLCDM]+)\s*$", "", text)

    # replace single newlines inside running text with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # restore paragraph spacing
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# Split a text into paragraphs based on blank lines
def split_paragraphs(text):
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return paras

# Split a paragraph into chunks of max 500 words
def chunk_paragraphs(paragraphs, max_chars=500):
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks

def process_all_pdfs():
    all_chunks = []
    chunk_id = 0

    for pdf_path in RAW_DIR.glob("*.pdf"):
        topic = pdf_path.stem.lower()   # abortion / hiv / infertility
        source = f"WHO {topic.capitalize()} Guideline"

        print(f"Processing {pdf_path.name}...")
        # split by page first
        pages = extract_pdf(pdf_path)
        # then split each page into paragraphs, and chunk those paragraphs
        for p in pages:
            page_no = p["page"]
            paragraphs = split_paragraphs(p["text"])
            page_chunks = chunk_paragraphs(paragraphs, max_chars=900)

            for chunk in page_chunks:
                all_chunks.append({
                    "chunk_id": f"{topic}_{chunk_id}",
                    "text": chunk,
                    "source": source,
                    "topic": topic,
                    "page": page_no
                })
                chunk_id += 1

    return all_chunks

if __name__ == "__main__":
    chunks = process_all_pdfs()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks → {OUT_FILE}")

    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print(len(data))
    print(data[0]["text"][:300])