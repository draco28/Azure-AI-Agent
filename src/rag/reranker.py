from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import numpy as np
from langsmith import traceable


def cosine_similarity(v1, v2) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@traceable
def rrf_fusion(results_list: list[list[tuple[Document, float]]], weights: list[float], k=60) -> list[tuple[Document,float]]:
    scores_dict = {}
    docs_dict = {}

    for (ranked_list, weight) in zip(results_list, weights):
        for rank, (doc, score) in enumerate(ranked_list, start=1):
            doc_key = (doc.metadata['source'], doc.metadata['chunk_index'])
            
            if doc_key not in scores_dict:
                scores_dict[doc_key] = 0.0
                docs_dict[doc_key] = doc
            
            scores_dict[doc_key] += weight * (1.0 / (rank + k))
    
    #convert dicts to sorted list of tuples
    results = [(docs_dict[key], score) for key, score in scores_dict.items()]
    return sorted(results, key=lambda x: x[1], reverse=True)

@traceable
def mmr_rerank(documents: list[tuple[Document, float]], embeddings: Embeddings, top_k: int = 4, lambda_param: float = 0.7) -> list[tuple[Document,float]]:
    texts = [doc.page_content for doc, score in documents]
    vectors = embeddings.embed_documents(texts)
    
    max_score = documents[0][1]
    min_score = documents[-1][1]

    score_range = max_score - min_score

    if score_range == 0:
        normalized_scores = [1.0 for _ in documents]
    else:
        normalized_scores = [(score - min_score) / score_range for _, score in documents]

    selected_indices = [0]
    remaining_indices = list(range(1, len(documents)))

    while len(selected_indices) < top_k and remaining_indices:
        best_index = None
        best_mmr = -float('inf')

        for idx in remaining_indices:
            relevance = normalized_scores[idx]
            similarity = max(cosine_similarity(vectors[s], vectors[idx]) for s in selected_indices)
            mmr = lambda_param * relevance - (1 - lambda_param) * similarity
            if mmr > best_mmr:
                best_mmr = mmr
                best_index = idx
        
        if best_index is not None:
            selected_indices.append(best_index)
            remaining_indices.remove(best_index)
    
    return [(documents[idx][0], documents[idx][1]) for idx in selected_indices]