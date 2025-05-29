from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from rag.embedding import load_chunks_from_csv
from rag.retrieval import load_embeddings, get_top_chunks
from rag.generation import format_prompt, query_llm
import os

# Model + load data
model = SentenceTransformer("all-mpnet-base-v2")
chunks = load_chunks_from_csv("data/pages_and_chunks.csv")
embeddings = load_embeddings("data/embeddings.npy")

# Test query
query = "What is Porter's Value Chain?"
top_chunks = get_top_chunks(query, chunks, embeddings, model, k=3)

# Prompt and LLM answer
prompt = format_prompt(query, top_chunks)
response = query_llm(prompt)

print(response)
