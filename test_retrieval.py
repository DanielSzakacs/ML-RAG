from sentence_transformers import SentenceTransformer
from rag.embedding import load_chunks_from_csv
from rag.retrieval import load_embeddings, get_top_chunks

embedding_path = "data/embeddings.npy"
csv_path = "data/pages_and_chunks.csv"
model = SentenceTransformer("all-mpnet-base-v2")

chunks = load_chunks_from_csv(csv_path)
embeddings = load_embeddings(embedding_path)

query = "What is the purpose of Porter's Value Chain?"
top_chunks = get_top_chunks(query, chunks, embeddings, model)

for chunk in top_chunks:
    print(f"\n--- Page {chunk['page_number']} ---")
    print(chunk["sentence_chunk"][:400], "...")
