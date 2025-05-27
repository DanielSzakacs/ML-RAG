import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_chunks_from_csv(csv_path: str):
    return pd.read_csv(csv_path).to_dict(orient="records")

def generate_embeddings(chunks: list[dict], model: SentenceTransformer, embedding_file: str):
    if os.path.exists(embedding_file):
        print(f"[INFO] Embedding fájl megtalálva: {embedding_file}, betöltés...")
        return np.load(embedding_file)
    
    print("[INFO] Embedding generálása...")
    embeddings = []
    for chunk in tqdm(chunks):
        emb = model.encode(chunk["sentence_chunk"], convert_to_tensor=False)
        embeddings.append(emb)
    
    embeddings_np = np.array(embeddings)
    np.save(embedding_file, embeddings_np)
    print(f"[INFO] Embedding elmentve: {embedding_file}")
    return embeddings_np
