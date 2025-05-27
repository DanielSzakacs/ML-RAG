import numpy as np 
import torch 
from sentence_transformers import SentenceTransformer, util

def load_embeddings(embedding_path: str) -> torch.Tensor: 
    """
        Load embedding and return it as Tensor
    """
    embeddings_np = np.load(embedding_path)
    return torch.tensor(embeddings_np, dtype=torch.float32)

def get_top_chunks(query: str, chunks: list[dict], embeddings: torch.Tensor, model: SentenceTransformer, k: int = 5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.dot_score(query_embedding, embeddings)[0]
    top_k_scores, top_k_indices  = torch.topk(scores, k=k)
    top_chunks = [chunks[i] for i in top_k_indices.tolist()]
    return top_chunks
