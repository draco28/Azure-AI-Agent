from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import get_settings, Settings


def load_document_type(file_path: str) -> list[Document]:
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".md"):
        return UnstructuredMarkdownLoader(file_path).load()
    else:
        raise ValueError(f"Unrecognized file type: {file_path}")

def generate_chunks(document: list[Document], settings: Settings | None = None) -> list[Document]:
    if settings is None:
        settings = get_settings()
    recursive_text_split = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
                                                          chunk_overlap=settings.chunk_overlap)
    chunks = recursive_text_split.split_documents(document)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks

