import fitz  # PyMuPDF
import os
import pandas as pd
from spacy.lang.en import English
from tqdm import tqdm
import re

nlp = English()
nlp.add_pipe("sentencizer")

def download_pdf(pdf_url: str, save_path: str):
    if not os.path.exists(save_path):
        import requests
        response = requests.get(pdf_url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"[INFO] PDF letöltve: {save_path}")
    else:
        print("[INFO] PDF már létezik")

def extract_text_chunks(pdf_path: str, number_of_chunk: int = 5):
    doc = fitz.open(pdf_path)
    pages_and_text = []

    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text().replace("\n", " ").strip()
        pages_and_text.append({
            "page_number": page_number + 1,
            "text": text,
            "sentences": [str(s) for s in nlp(text).sents]
        })

    all_chunks = []
    for page in pages_and_text:
        sentences = page["sentences"]
        chunks = [sentences[i:i + number_of_chunk] for i in range(0, len(sentences), number_of_chunk)]
        for chunk in chunks:
            text_chunk = " ".join(chunk)
            text_chunk = re.sub(r"\.([A-Z])", r". \1", text_chunk)
            all_chunks.append({
                "page_number": page["page_number"],
                "sentence_chunk": text_chunk,
                "chunk_char_count": len(text_chunk),
                "chunk_word_count": len(text_chunk.split()),
                "chunk_token_count": len(text_chunk) / 4
            })

    return pd.DataFrame(all_chunks)