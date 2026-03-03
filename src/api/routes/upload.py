import os
from fastapi import APIRouter, UploadFile, File

from src.config import get_settings
from src.rag.ingest import load_document_type, generate_chunks
from src.rag.embeddings import create_embeddings
from src.rag.store import create_index, save_index, load_index
from src.rag.azure_search import upload_documents
from src.api.schema import UploadResponse

router = APIRouter(prefix="/api")

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()
    
    save_path = os.path.join("data/documents", file.filename)
    os.makedirs("data/documents", exist_ok=True)
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    docs = load_document_type(save_path)
    chunks = generate_chunks(docs)

    embeddings = create_embeddings(settings)
    if settings.search_backend == "azure":
        upload_documents(chunks, embeddings, settings)
    elif settings.search_backend == "faiss":
        if os.path.exists(settings.vector_store_path):
            index = load_index(settings.vector_store_path, embeddings)
        else:
            index = create_index(chunks, embeddings)
            save_index(index, settings.vector_store_path)
            return UploadResponse(
                filename=file.filename,
                chunks=len(chunks)
            )

        index.add_documents(chunks)
        save_index(index, settings.vector_store_path)

    return UploadResponse(
        filename=file.filename,
        chunks=len(chunks)
    )