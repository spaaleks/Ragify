from __future__ import annotations
from datetime import datetime, UTC
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy import Column, String, BigInteger, Integer, DateTime, Text, UniqueConstraint, JSON, ForeignKey, Enum
from pgvector.sqlalchemy import Vector
from enum import Enum as PyEnum

Base = declarative_base()

class FileStatus(str, PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# File model: add fields
class File(Base):
    __tablename__ = "files"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True, index=True, nullable=False)
    mtime_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime: Mapped[str] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")   # pending|processing|success|failed|skipped
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    chunks = relationship("Chunk", cascade="all, delete-orphan", back_populates="file")
class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    file = relationship("File", back_populates="chunks")

    __table_args__ = (UniqueConstraint("file_id", "chunk_index", name="uq_chunk_file_idx"),)

class ChatTurn(Base):
    __tablename__ = "chat_history"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(128), index=True, nullable=False)
    folder = Column(String(128), index=True, nullable=False)
    ts = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)  
    role = Column(String(16), nullable=False)     # 'user' | 'assistant'
    content = Column(Text, nullable=False)
