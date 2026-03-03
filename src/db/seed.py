from sqlalchemy import select
from src.db.models import FileRecord


sample_files = [
    FileRecord(
        file_id="DOC-2024-0847",
        filename="quarterly_report_q4_2024.md",
        status="approved",
        submitted_by="Sarah Chen",
        department="Finance",
    ),
    FileRecord(
        file_id="DOC-2024-0923",
        filename="employee_handbook.txt",
        status="approved",
        submitted_by="James Morton",
        updated_by="Lisa Park",
        department="Human Resources",
    ),
    FileRecord(
        file_id="DOC-2024-1015",
        filename="it_security_policy.md",
        status="processing",
        submitted_by="Raj Patel",
        department="IT Security",
    ),
    FileRecord(
        file_id="DOC-2025-0042",
        filename="product_faq.txt",
        status="pending",
        submitted_by="Emily Rodriguez",
        department="Product",
    ),
    FileRecord(
        file_id="DOC-2025-0078",
        filename="q1_2025_budget_proposal.pdf",
        status="rejected",
        submitted_by="Michael Tanaka",
        updated_by="Sarah Chen",
        department="Finance",
    ),
    FileRecord(
        file_id="DOC-2025-0103",
        filename="engineering_onboarding_guide.md",
        status="processing",
        submitted_by="Dr. Sarah Zhang",
        department="Engineering",
    ),
]


async def seed_database(session_factory):
    async with session_factory() as session:
        result = await session.execute(select(FileRecord).limit(1))
        if result.scalar() is not None:
            return

        session.add_all(sample_files)
        await session.commit()