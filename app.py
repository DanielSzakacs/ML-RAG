import gradio as gr
from sentence_transformers import SentenceTransformer
from rag.embedding import load_chunks_from_csv, generate_embeddings
from rag.retrieval import load_embeddings, get_top_chunks
from rag.generation import format_prompt, query_llm
from rag.data import download_pdf, extract_text_chunks
import fitz
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

def load_source_data():
    pdf_url = "https://raw.githubusercontent.com/DanielSzakacs/RAG-demo-v1/main/source/businessAnalysis.pdf"
    csv_path = "data/pages_and_chunks.csv"
    pdf_path = "data/businessAnalysis.pdf"
    embedding_path = "data/embeddings.npy"

    # Ensure 'data/' directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("[INFO] 'data/' folder created.")

    # Load source data if not exist
    download_pdf(pdf_url, pdf_path)
    if not os.path.exists(csv_path):
        print("[INFO] Extracting chunks from PDF...")
        df = extract_text_chunks(pdf_path)
        df.to_csv(csv_path, index=False)
    else:
        print("[INFO] CSV already exist")


    if not os.path.exists(embedding_path):
        model = SentenceTransformer("all-mpnet-base-v2")
        chunks = load_chunks_from_csv(csv_path)
        generate_embeddings(chunks, model, embedding_path)
load_source_data()

# Load
model = SentenceTransformer("all-mpnet-base-v2")
chunks = load_chunks_from_csv("data/pages_and_chunks.csv")
embeddings = load_embeddings("data/embeddings.npy")
pdf_path = "data/businessAnalysis.pdf"

# Return PDF page
def get_page_image(page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img

# Gradio callback
def rag_ask(query):
    top_chunks = get_top_chunks(query, chunks, embeddings, model)
    prompt = format_prompt(query, top_chunks)
    answer = query_llm(prompt)
    page_img = get_page_image(top_chunks[0]["page_number"])
    return answer, page_img

# Gradio UI
demo = gr.Interface(
    fn=rag_ask,
    inputs="text",
    outputs=["text", "image"],
    title="📚 RAG Demo: Business Analysis",
    description="Ask a question from the PDF and you'll get the answer + the source page.",
    examples=[
        "What is Porter's Value Chain?",
        "What are value propositions?",
        "What factors influence competition?"
    ]
)

if __name__ == "__main__":
    demo.launch()
