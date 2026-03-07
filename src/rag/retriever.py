"""
Hybrid and Azure Search Retriever implementations for RAG (Retrieval-Augmented Generation).

This module provides two retriever classes that combine different search strategies
to improve document retrieval quality:
- HybridRetriever: Combines dense (FAISS) and sparse (BM25) retrieval with RRF fusion
- AzureSearchRetriever: Uses Azure Cognitive Search with hybrid search capabilities

Both retrievers apply MMR (Maximal Marginal Relevance) reranking to reduce redundancy
and improve diversity in the final results.
"""

import logging
import time
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langsmith import traceable

from src.rag.azure_search import hybrid_search
from src.rag.sparse import BM25Retriever
from src.rag.reranker import mmr_rerank, rrf_fusion
from src.config import get_settings, Settings

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        """Retrieve relevant documents for a given query."""
        ...
    
    @traceable
    def format_citation(self, results: list[tuple[Document, float]]) -> str:
        """
        Format retrieved documents with source citations.
        
        Creates a human-readable string with document content and metadata
        for use in LLM context or display to users.
        
        Args:
            results: List of (Document, score) tuples from retrieval
            
        Returns:
            Formatted string with documents separated by double newlines,
            each including source file, chunk index, and relevance score
        """
        formatted = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            chunk_index = doc.metadata.get("chunk_index", "N/A")
            text = doc.page_content
            formatted.append(
                f"{text}\n[Source: {source}, chunk {chunk_index}, score {score:.2f}]"
            )
        return "\n\n".join(formatted)


class HybridRetriever(BaseRetriever):
    """
    A hybrid retriever that combines dense and sparse retrieval strategies.
    
    This retriever uses both FAISS (dense vector similarity) and BM25 (sparse lexical matching)
    to find relevant documents. Results from both methods are fused using Reciprocal Rank Fusion
    (RRF) and then reranked using Maximal Marginal Relevance (MMR) to balance relevance
    and diversity.
    
    Attributes:
        faiss_index: FAISS vector store for dense retrieval
        bm25_retriever: BM25 retriever for sparse lexical matching
        embeddings: Embedding model for vector operations
        settings: Application settings configuration
    
    Example:
        >>> retriever = HybridRetriever(faiss_index, bm25_retriever, embeddings)
        >>> results = retriever.retrieve("What is the refund policy?", top_k=5)
        >>> citation_text = retriever.format_citation(results)
    """
    
    # Weights for RRF fusion: dense retrieval is weighted higher than sparse
    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3
    
    def __init__(
        self, 
        faiss_index: FAISS, 
        bm25_retriever: BM25Retriever, 
        embeddings: Embeddings, 
        settings: Settings | None = None
    ):
        """
        Initialize the hybrid retriever with dense and sparse components.
        
        Args:
            faiss_index: Pre-built FAISS index containing document embeddings
            bm25_retriever: Initialized BM25 retriever with tokenized corpus
            embeddings: Embedding model for similarity computations
            settings: Optional settings object; loads from environment if None
        """
        if settings is None:
            settings = get_settings()
        
        self.faiss_index = faiss_index
        self.bm25_retriever = bm25_retriever
        self.embeddings = embeddings
        self.settings = settings

    @traceable
    def retrieve(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        """
        Retrieve relevant documents using hybrid dense-sparse search.
        
        The retrieval pipeline:
        1. Dense retrieval via FAISS similarity search
        2. Sparse retrieval via BM25 lexical matching
        3. RRF fusion to combine both result sets
        4. MMR reranking to reduce redundancy
        
        Args:
            query: The search query string
            top_k: Number of documents to return (default: 4)
            
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        start_time = time.perf_counter()
        
        # Step 1: Dense retrieval using FAISS
        # FAISS returns L2 distance; convert to similarity score using 1/(1+distance)
        dense_raw = self.faiss_index.similarity_search_with_score(query, k=top_k)
        dense_results = [(doc, 1.0 / (1.0 + score)) for doc, score in dense_raw]

        # Step 2: Sparse retrieval using BM25
        sparse_results = self.bm25_retriever.search(query, top_k=top_k)
        
        # Step 3: Fuse results using Reciprocal Rank Fusion
        # Dense results are weighted higher (0.7) than sparse (0.3)
        fused_results = rrf_fusion(
            [dense_results, sparse_results], 
            weights=[self.DENSE_WEIGHT, self.SPARSE_WEIGHT]
        )

        # Step 4: Apply MMR reranking for diversity
        reranked_results = mmr_rerank(fused_results, self.embeddings, top_k=top_k)

        # Log retrieval metrics for monitoring
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "tool.rag.retrieval",
            extra={
                "search_backend": "faiss+bm25",
                "top_k": top_k,
                "doc_count": len(reranked_results),
                "latency_ms": latency_ms,
            },
        )

        return reranked_results

    


class AzureSearchRetriever(BaseRetriever):
    """
    A retriever that uses Azure Cognitive Search for hybrid retrieval.
    
    This retriever leverages Azure's built-in hybrid search capabilities,
    combining vector similarity and keyword matching in a single query.
    Results are then reranked using MMR for improved diversity.
    
    Attributes:
        embeddings: Embedding model for query vectorization
        settings: Application settings with Azure Search configuration
    
    Example:
        >>> retriever = AzureSearchRetriever(embeddings, settings)
        >>> results = retriever.retrieve("How do I reset my password?", top_k=5)
    """
    
    def __init__(self, embeddings: Embeddings, settings: Settings | None = None):
        """
        Initialize the Azure Search retriever.
        
        Args:
            embeddings: Embedding model for query vectorization
            settings: Optional settings object with Azure Search credentials;
                      loads from environment if None
        """
        if settings is None:
            settings = get_settings()
        
        self.embeddings = embeddings
        self.settings = settings

    @traceable
    def retrieve(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        """
        Retrieve relevant documents using Azure Cognitive Search.
        
        The retrieval pipeline:
        1. Hybrid search via Azure (combines vector + keyword search)
        2. MMR reranking to reduce redundancy
        
        Args:
            query: The search query string
            top_k: Number of documents to return (default: 4)
            
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        start_time = time.perf_counter()
        
        # Step 1: Execute hybrid search on Azure Cognitive Search
        fused_results = hybrid_search(
            query, 
            top_k=top_k, 
            embeddings=self.embeddings, 
            settings=self.settings
        )

        # Step 2: Apply MMR reranking for diversity
        reranked_results = mmr_rerank(fused_results, self.embeddings, top_k=top_k)
        
        # Log retrieval metrics for monitoring
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "tool.rag.retrieval",
            extra={
                "search_backend": "azure_search",
                "top_k": top_k,
                "doc_count": len(reranked_results),
                "latency_ms": latency_ms,
            },
        )
        
        return reranked_results


