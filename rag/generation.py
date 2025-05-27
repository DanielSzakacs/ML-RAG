import os
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def format_prompt(query: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        [f"[Page {chunk['page_number']}]\n{chunk['sentence_chunk']}" for chunk in context_chunks]
    )

    prompt = f"""You are a helpful AI assistant. 
Use the following extracted document passages to answer the user's question. 
Base your answer only on the provided context.

Context:
{context_text}

Question: {query}
Answer:"""

    return prompt



def query_llm(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Failed LLM request: {response.text}")
    generated = response.json()
    return generated[0]["generated_text"]
