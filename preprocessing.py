import json
import re
import unicodedata

# text normalization
def normalize_text(text):
    # normalise the text to NFKC form to standardize characters
    text = unicodedata.normalize("NFKC", text)

    #clean the unicode text by replacing common unicode characters with their ASCII equivalents
    replacements = {
        "\u00a0": " ",   # non-breaking space
        "\u00ad": "",    # soft hyphen
        "ﬁ": "fi",
        "ﬂ": "fl",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # remove repeated headers/footers
    text = re.sub(r"(?m)^(.*?)(\s*\1)+\s*$", r"\1", text)
    
    # remove odd invisible/control chars except newline/tab
    text = re.sub(r"[^\S\n\t]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)

    return text.strip()

# lowercase and remove punctuation
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

if __name__ == "__main__":
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    normalized = normalize_text(sample_text)
    cleaned = clean_text(normalized)
    print("Original:", sample_text)
    print("Normalized:", normalized)
    print("Cleaned:", cleaned)