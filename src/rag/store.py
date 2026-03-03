from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

import os

def create_index(chunks: list[Document], embeddings:Embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)

def save_index(index: FAISS, path: str):
    index.save_local(path)

def load_index(path: str, embeddings: Embeddings) -> FAISS:
    if not os.path.exists(path):
        raise FileNotFoundError(f"no index found at {path}")
    else:
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
