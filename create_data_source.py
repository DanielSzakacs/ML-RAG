from rag.embedding import load_chunks_from_csv, generate_embeddings
from rag.data import download_pdf, extract_text_chunks
from sentence_transformers import SentenceTransformer
import os

pdf_url = "https://raw.githubusercontent.com/DanielSzakacs/RAG-demo-v1/main/source/businessAnalysis.pdf"
pdf_path = "data/businessAnalysis.pdf"
csv_path = "data/pages_and_chunks.csv"
embedding_path = "data/embeddings.npy"

# 1. Download PDF
download_pdf(pdf_url, pdf_path)

# 2. Processing and saving CSV if it does not already exist
if not os.path.exists(csv_path):
    print("[INFO] Extracting chunks from PDF...")
    df = extract_text_chunks(pdf_path)
    df.to_csv(csv_path, index=False)
else:
    print("[INFO] CSV már létezik")

# 3. Embedding generálás vagy betöltés
model = SentenceTransformer("all-mpnet-base-v2")
chunks = load_chunks_from_csv(csv_path)
embeddings = generate_embeddings(chunks, model, embedding_path)

print("✔️ Teszt sikeres! Embedding shape:", embeddings.shape)