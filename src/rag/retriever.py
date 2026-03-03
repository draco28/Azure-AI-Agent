from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langsmith import traceable

from src.rag.azure_search import hybrid_search
from src.rag.sparse import BM25Retriever
from src.rag.reranker import mmr_rerank, rrf_fusion
from src.config import get_settings, Settings

class HybridRetriever:
    def __init__(self, faiss_index: FAISS, bm25_retriever: BM25Retriever, embeddings: Embeddings, settings: Settings | None = None):
        if settings is None:
            settings = get_settings()
        self.faiss_index = faiss_index
        self.bm25_retriever = bm25_retriever
        self.embeddings = embeddings
        self.settings = settings

    @traceable
    def retrieve(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        dense_raw = self.faiss_index.similarity_search_with_score(query, k=top_k)
        dense_results = [(doc, 1.0/ (1.0 + score)) for doc, score in dense_raw]

        sparse_results = self.bm25_retriever.search(query, top_k=top_k)
        
        fused_results = rrf_fusion([dense_results, sparse_results], weights=[0.7, 0.3])

        reranked_results = mmr_rerank(fused_results, self.embeddings, top_k=top_k)

        return reranked_results

    @traceable
    def format_citation(self, results: list[tuple[Document, float]]) -> str:
        formatted = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            chunk_index = doc.metadata.get("chunk_index", "N/A")
            text = doc.page_content
            formatted.append(f"{text}\n[Source: {source}, chunk {chunk_index}, score {score:.2f}]")
        return "\n\n".join(formatted)


class AzureSearchRetriever():
    def __init__(self,embeddings: Embeddings, settings: Settings | None = None):
        if settings is None:
            settings = get_settings()
        self.embeddings = embeddings
        self.settings = settings

    @traceable
    def retrieve(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        fused_results = hybrid_search(query, top_k=top_k,embeddings=self.embeddings, settings=self.settings)

        reranked_results = mmr_rerank(fused_results, self.embeddings, top_k=top_k)
        return reranked_results

    @traceable
    def format_citation(self, results: list[tuple[Document, float]]) -> str:
        formatted = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            chunk_index = doc.metadata.get("chunk_index", "N/A")
            text = doc.page_content
            formatted.append(f"{text}\n[Source: {source}, chunk {chunk_index}, score {score:.2f}]")
        return "\n\n".join(formatted)

