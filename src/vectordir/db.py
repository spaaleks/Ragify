from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import DBConfig


def _default_vector_dim() -> int:
    try:
        env_dim = int(os.getenv("VECTOR_EMBED_DIM", "768") or "768")
        if env_dim > 0:
            return env_dim
    except (TypeError, ValueError):
        pass
    return 768


def ensure_match_documents_function(conn, dim: Any | None = None) -> None:
    try:
        resolved = int(dim) if dim is not None else _default_vector_dim()
        if resolved <= 0:
            resolved = _default_vector_dim()
    except (TypeError, ValueError):
        resolved = _default_vector_dim()

    # ensure consistent signature before recreation
    conn.exec_driver_sql("DROP FUNCTION IF EXISTS match_documents(vector, int, jsonb);")

    sql = f"""
    CREATE FUNCTION match_documents (
        query_embedding vector({resolved}),
        match_count int DEFAULT NULL,
        filter jsonb DEFAULT '{{}}'::jsonb
    )
    RETURNS TABLE (
        id bigint,
        content text,
        metadata jsonb,
        embedding jsonb,
        similarity double precision
    )
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN QUERY
        SELECT
            c.id::bigint,
            c.content,
            c.metadata,
            (c.embedding::text)::jsonb AS embedding,
            1 - (c.embedding <=> query_embedding) AS similarity
        FROM chunks c
        WHERE (filter IS NULL OR filter = '{{}}'::jsonb OR c.metadata @> filter)
        ORDER BY c.embedding <=> query_embedding
        LIMIT COALESCE(match_count, 2147483647);
    END;
    $$;
    """
    conn.exec_driver_sql(sql)

def make_dsn(db: DBConfig) -> str:
    return f"postgresql+psycopg://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"

def make_engine(db: DBConfig):
    dsn = make_dsn(db)
    return create_engine(dsn, pool_pre_ping=True, pool_size=5, max_overflow=10)

def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)

@contextmanager
def session_scope(Session):
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
