from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    #LLM
    glm_api_key: SecretStr
    glm_base_url: str = "https://api.z.ai/api/coding/paas/v4"
    chat_model: str = "glm-4.7"

    #Embeddings
    azure_openai_endpoint: str
    azure_openai_api_key: SecretStr
    azure_openai_api_version: str
    embedding_model: str = "text-embedding-3-small"
    embedding_deployment: str

    #RAG
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 4
    vector_store_path: str = ("./data/vector_store")

    #PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5434
    postgres_db: str = "azure_ai_agent"
    postgres_user: str = "postgres"
    postgres_password: SecretStr

    #AzureAISearch
    azure_search_endpoint: str
    azure_search_api_key: SecretStr
    azure_search_index_name: str = "documents"

    #Vector Store
    search_backend: str = "faiss" #"faiss" or "azure"

    #Redis
    redis_url: str = "redis://localhost:6381"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache(maxsize=None)
def get_settings():
    return Settings()