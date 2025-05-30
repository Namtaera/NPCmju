# rag.py

import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")
    return OpenAI(api_key=api_key)

def chunk_text(text, max_chars=500):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def build_faiss(chunks, model="text-embedding-ada-002"):
    client = get_openai_client()
    embeddings = [
        client.embeddings.create(input=chunk, model=model).data[0].embedding
        for chunk in chunks
    ]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    return index, embeddings

def retrieve_top_k(index, query, chunks, model="text-embedding-ada-002", k=3):
    client = get_openai_client()
    q_emb = client.embeddings.create(input=query, model=model).data[0].embedding
    D, I = index.search(np.array([q_emb], dtype="float32"), k)
    return [chunks[i] for i in I[0]]
