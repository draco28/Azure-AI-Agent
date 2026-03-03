from langchain_openai import AzureOpenAIEmbeddings

from src.config import get_settings, Settings

def create_embeddings(settings: Settings | None = None) -> AzureOpenAIEmbeddings:
    if settings is None:
        settings = get_settings()

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=settings.embedding_deployment,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        model= settings.embedding_model
    )

    return embeddings