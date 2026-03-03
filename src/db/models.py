from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional

class Base(DeclarativeBase):
    pass



class FileRecord(Base):
    __tablename__ = "files"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[str] = mapped_column(String(50), unique=True)
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50))
    submitted_by: Mapped[str] = mapped_column(String(255))
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    department: Mapped[str] = mapped_column(String(255))

class FileStatusResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    file_id: str
    filename: str
    status: str
    submitted_by: str
    submitted_at: datetime
    updated_by: Optional[str]
    updated_at: datetime
    department: str