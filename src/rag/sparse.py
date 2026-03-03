from langchain_core.documents import Document
from langsmith import traceable
from rank_bm25 import BM25Okapi
import numpy as np


class BM25Retriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.tokenizer_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenizer_corpus)

    @traceable
    def search(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_k_indices]