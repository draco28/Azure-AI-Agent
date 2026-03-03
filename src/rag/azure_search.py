import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)

from src.config import get_settings,Settings



def create_search_index(settings: Settings | None = None) -> SearchIndex | None:
    if settings is None:
        settings = get_settings()

    index_client = SearchIndexClient(endpoint=settings.azure_search_endpoint,
                                     credential=AzureKeyCredential(settings.azure_search_api_key.get_secret_value()))

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, retrievable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=1536,
                    vector_search_profile_name="vector-profile"),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, retrievable=True)
    ]
    vector_search = VectorSearch(algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
                                 profiles=[VectorSearchProfile(name="vector-profile",
                                                               algorithm_configuration_name="hnsw-config")])
    index = SearchIndex(name=settings.azure_search_index_name, fields=fields, vector_search=vector_search)

    index_client.create_or_update_index(index)
    return index

def upload_documents(chunks: list[Document], embeddings: Embeddings, settings: Settings | None = None) -> None:
    if settings is None:
        settings = get_settings()

    search_client = SearchClient(endpoint=settings.azure_search_endpoint,
                                 index_name=settings.azure_search_index_name,
                                 credential=AzureKeyCredential(settings.azure_search_api_key.get_secret_value()))

    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    documents = []

    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        metadata = chunk.metadata or {}

        source = metadata.get("source","unknown")
        chunk_index = metadata.get("chunk_index", i)

        raw_id = f"{source}_{chunk_index}"
        unique_id = hashlib.md5(raw_id.encode()).hexdigest()

        doc = {
            "id": unique_id,
            "content": content,
            "content_vector": vectors[i],
            "source": source,
            "chunk_index": chunk_index
        }

        documents.append(doc)

    result = search_client.upload_documents(documents=documents)

def hybrid_search(query: str,top_k: int,embeddings: Embeddings,settings: Settings | None = None) -> list[tuple[Document, float]]:
    if settings is None:
        settings = get_settings()
    search_client = SearchClient(endpoint=settings.azure_search_endpoint,
                                 index_name=settings.azure_search_index_name,
                                 credential=AzureKeyCredential(settings.azure_search_api_key.get_secret_value()))
    query_vector = embeddings.embed_query(query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=top_k
    )

    documents = []
    for result in results:
        doc = Document(
            page_content=result["content"],
            metadata={"source": result["source"], "chunk_index": result["chunk_index"]}
        )
        documents.append((doc,result["@search.score"]))

    return documents