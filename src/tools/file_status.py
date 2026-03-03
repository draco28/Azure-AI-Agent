import logging
import time

from langchain_core.tools import tool
from sqlalchemy import select
from src.db.models import FileRecord

logger = logging.getLogger(__name__)

async def query_files(session_factory, filename=None, file_id=None, status=None) -> str:
    start_time = time.perf_counter()

    query = select(FileRecord)
    if filename:
        for word in filename.split():
            query = query.where(FileRecord.filename.ilike(f"%{word}%"))
    if file_id:
        query = query.where(FileRecord.file_id == file_id)
    if status:
        query = query.where(FileRecord.status == status)
    async with session_factory() as session:
        result = await session.execute(query)
        files = result.scalars().all()

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    filters = {k: v for k, v in {"filename": filename, "file_id": file_id, "status": status}.items() if v}
    logger.info(
        "tool.file_status.query",
        extra={
            "filters": filters,
            "results_count": len(files),
            "latency_ms": latency_ms,
            "success": True,
        },
    )

    if not files:
        return "No files found matching the given criteria"
    formatted = []
    for file in files:
        formatted.append(f"""File: {file.filename}
ID: {file.file_id}
Status: {file.status}
Submitted by: {file.submitted_by}
Department: {file.department}""")
    return "\n".join(formatted)



def create_file_status_tool(session_factory):
    @tool
    async def file_status_tool(filename: str | None = None, file_id: str | None = None, status: str | None = None) -> str:
        """
       Get the status of a file.

       Args:
           filename: The name of the file (optional if file_id is provided)
           file_id: The ID of the file (optional if filename is provided)
           status: The status to filter by (optional, if not provided, returns current status)

       Returns:
           The current status of the file
           """
        if not filename and not file_id:
            return "Please provide either a filename or file_id"
        return await query_files(session_factory, filename, file_id, status)
    return file_status_tool

    