from __future__ import annotations
from pathlib import Path
import os, importlib.util
from typing import List, Tuple
from tqdm import tqdm
import logging
from datetime import datetime, UTC

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document

from .config import FolderAdapter, AppConfig, S3Config
from .ollama import resolve_base_url as _ollama_base_url, build_headers as _ollama_headers
from .db import make_engine, make_session_factory, session_scope, ensure_match_documents_function
from .models import Base, File, Chunk, FileStatus
from .loaders import discover_paths, load_file, split_docs, guess_mime

logger = logging.getLogger("vectordir")

def _ensure_schema(engine, embedding_dim: int):
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        ensure_match_documents_function(conn, embedding_dim)

def _embedding_client(model_name: str, folder: FolderAdapter, appcfg: AppConfig):
    m = appcfg.models[model_name]
    if m.provider == "openai":
        return OpenAIEmbeddings(
            model=model_name,
            api_key=folder.openai.api_key or os.getenv("OPENAI_API_KEY", ""),
            openai_api_base=folder.openai.base_url or appcfg.global_openai.base_url,
        )
    if m.provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=m.api_key or os.getenv("GEMINI_API_KEY"),
        )
    if m.provider == "vertex_ai":
        return VertexAIEmbeddings(
            model_name=model_name,
            project=m.project,
            location=m.location,
        )
    if m.provider == "ollama":
        headers = _ollama_headers(m)
        kwargs = {
            "model": model_name,
            "base_url": _ollama_base_url(m),
        }
        if headers:
            kwargs["headers"] = headers
        return OllamaEmbeddings(**kwargs)
    raise ValueError(f"unsupported provider: {m.provider}")

def _stat_file(p: Path) -> Tuple[int, int]:
    st = p.stat()
    return (st.st_mtime_ns, st.st_size)

def _needs_reindex(existing: File, mtime_ns: int, size_bytes: int) -> bool:
    if existing.mtime_ns != mtime_ns or existing.size_bytes != size_bytes:
        return True
    # If we have processed_at, ensure processing happened after the file mtime
    if existing.processed_at is None:
        return True
    processed_at = existing.processed_at
    if processed_at.tzinfo is None:
        processed_at = processed_at.replace(tzinfo=UTC)
    else:
        processed_at = processed_at.astimezone(UTC)
    mtime_dt = datetime.fromtimestamp(mtime_ns / 1e9, tz=UTC)
    return processed_at < mtime_dt

def _upsert_file(session: Session, path: Path, mtime_ns: int, size_bytes: int, mime: str) -> File:
    existing = session.scalar(select(File).where(File.path == str(path)))
    if existing:
        if _needs_reindex(existing, mtime_ns, size_bytes):
            # file changed or was never processed or processed before last mtime
            session.execute(delete(Chunk).where(Chunk.file_id == existing.id))
            existing.mtime_ns = mtime_ns
            existing.size_bytes = size_bytes
            existing.mime = mime
            existing.status = "pending"
            # keep retry_count as-is
            session.add(existing)
        return existing
    f = File(path=str(path), mtime_ns=mtime_ns, size_bytes=size_bytes, mime=mime, status="pending", retry_count=0)
    session.add(f)
    session.flush()
    return f


def _load_tag_generator(spec_str: str):
    file_path, _, class_name = spec_str.partition(":")
    spec = importlib.util.spec_from_file_location("tagger", file_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls()

def _line_ranges_for_chunks(full_text: str, chunks: List[str]) -> List[Tuple[int, int]]:
    lines = full_text.splitlines()
    joined = "\n".join(lines)
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        idx = joined.find(chunk, cursor)
        if idx == -1:
            from_line = joined[:cursor].count("\n") + 1
            to_line = from_line + chunk.count("\n")
            ranges.append((from_line, to_line))
            cursor += len(chunk)
            continue
        from_line = joined[:idx].count("\n") + 1
        to_line = from_line + chunk.count("\n")
        ranges.append((from_line, to_line))
        cursor = idx + len(chunk)
    return ranges

def _effective_s3(folder: FolderAdapter, appcfg: AppConfig) -> S3Config | None:
    return folder.s3 if folder.s3 else appcfg.s3

def index_folder_adapter(
    folder: FolderAdapter,
    appcfg: AppConfig,
    force_reindex: bool = False,
    max_files: int | None = None,
):
    root = Path(folder.path).expanduser().resolve()

    # resolve embedding model + dim safely (dim may be absent for chat-only models)
    model_name = folder.embedding_model or appcfg.global_embedding_model
    if not model_name:
        raise ValueError(f"No embedding_model for folder {folder.name} and no global default")
    m = appcfg.models.get(model_name)
    dim = int(getattr(m, "dim", 768) or 768)
    os.environ["VECTOR_EMBED_DIM"] = str(dim)

    # infra
    engine = make_engine(folder.db)
    _ensure_schema(engine, dim)
    Session = make_session_factory(engine)

    embeddings = _embedding_client(model_name, folder, appcfg)
    tagger = _load_tag_generator(folder.tag_generator) if folder.tag_generator else None
    s3cfg = _effective_s3(folder, appcfg)
    max_retries = folder.max_retries or appcfg.max_retries

    # files to scan
    file_iter = list(discover_paths(root))
    if max_files:
        file_iter = file_iter[:max_files]

    with session_scope(Session) as session:
        for p in tqdm(file_iter, desc=f"Indexing {folder.name}"):
            mtime_ns, size_bytes = _stat_file(p)
            mime = guess_mime(p)

            # fetch existing
            existing = session.scalar(select(File).where(File.path == str(p)))
            needs_reindex = True
            if existing and not force_reindex:
                m_changed = (existing.mtime_ns != mtime_ns or existing.size_bytes != size_bytes)
                if not m_changed and existing.processed_at:
                    # normalize tz
                    pa = existing.processed_at
                    if pa.tzinfo is None:
                        pa = pa.replace(tzinfo=UTC)
                    else:
                        pa = pa.astimezone(UTC)
                    mtime_dt = datetime.fromtimestamp(mtime_ns / 1e9, tz=UTC)
                    needs_reindex = pa < mtime_dt
                else:
                    needs_reindex = True
                if not m_changed and existing.processed_at and not needs_reindex:
                    # mark success and skip
                    if getattr(existing, "status", None) != FileStatus.SUCCESS:
                        existing.status = FileStatus.SUCCESS
                        existing.processed_at = existing.processed_at or datetime.now(UTC)
                        existing.last_error = None
                        session.add(existing)
                        session.commit()
                    continue

            # upsert (also resets status to pending when metadata changed)
            file_row = _upsert_file(session, p, mtime_ns, size_bytes, mime)

            # retry guard
            if getattr(file_row, "retry_count", 0) >= max_retries:
                file_row.status = FileStatus.SKIPPED
                file_row.last_attempt_at = datetime.now(UTC)
                session.add(file_row)
                session.commit()
                logger.warning("Skipping %s (retries=%d >= max=%d)", p, file_row.retry_count, max_retries)
                continue

            # mark processing
            file_row.status = FileStatus.PROCESSING
            file_row.last_attempt_at = datetime.now(UTC)
            file_row.last_error = None
            session.add(file_row)
            session.commit()

            try:
                # load
                docs = load_file(p, appcfg, s3cfg)
                if not docs:
                    raise RuntimeError("no docs returned by loader")

                # chunking config
                chunk_cfg = folder.chunking or appcfg.chunking
                chunks_docs: List[Document] = split_docs(
                    docs,
                    chunk_size=chunk_cfg.size,
                    chunk_overlap=chunk_cfg.overlap,
                )
                texts: List[str] = [c.page_content for c in chunks_docs]
                if not texts:
                    raise RuntimeError("no text after splitting")
                logger.info(
                    "Index: %s chunks=%d mime=%s size=%d overlap=%d",
                    p.name, len(texts), mime, chunk_cfg.size, chunk_cfg.overlap
                )

                # compute line ranges against best-effort full text
                try:
                    full_text = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    full_text = "\n".join(texts)
                line_ranges = _line_ranges_for_chunks(full_text, texts)

                # embed
                vectors = embeddings.embed_documents(texts)
                logger.info("Embed: %s vectors=%d model=%s dim=%d", p.name, len(vectors), model_name, dim)

                # tags from tagger (pass relative path only)
                tags: List[str] = []
                if tagger:
                    try:
                        rel_path = p.relative_to(root)
                    except Exception:
                        rel_path = p.name  # fallback
                    try:
                        tags = tagger.generate(
                            rel_path,
                            {"mtime_ns": mtime_ns, "size_bytes": size_bytes, "mime": mime},
                        ) or []
                    except Exception:
                        tags = []

                # purge old chunks if reprocessing
                if existing:
                    session.execute(delete(Chunk).where(Chunk.file_id == file_row.id))
                    session.flush()

                # insert chunks
                for idx, (doc, vec, (from_line, to_line)) in enumerate(zip(chunks_docs, vectors, line_ranges)):
                    meta = {
                        "loc": {"lines": {"from": from_line, "to": to_line}},
                        "tags": tags,
                        "blobType": mime,
                    }
                    session.add(
                        Chunk(
                            file_id=file_row.id,
                            chunk_index=idx,
                            content=doc.page_content,
                            meta=meta,
                            embedding=vec,
                        )
                    )

                # success
                file_row.status = FileStatus.SUCCESS
                file_row.processed_at = datetime.now(UTC)
                file_row.last_error = None
                session.add(file_row)
                session.commit()
                logger.info("DB: upserted %s chunks=%d", p.name, len(texts))

            except Exception as e:
                # failure with retry increment
                file_row.retry_count = (getattr(file_row, "retry_count", 0) + 1)
                file_row.status = FileStatus.FAILED
                file_row.last_error = f"{type(e).__name__}: {e}"
                file_row.last_attempt_at = datetime.now(UTC)
                session.add(file_row)
                session.commit()
                tqdm.write(f"[process] {p}: {type(e).__name__}: {e}")
                logger.exception("Process failed for %s", p)
                continue

def vacuum_deleted_files(folder: FolderAdapter, appcfg: AppConfig) -> int:
    """
    Remove DB entries for files that no longer exist on disk.
    Returns number of files removed.
    """
    engine = make_engine(folder.db)
    Session = make_session_factory(engine)
    removed = 0
    with session_scope(Session) as session:
        rows = session.query(File).all()
        for r in rows:
            if not Path(r.path).exists():
                session.execute(delete(Chunk).where(Chunk.file_id == r.id))
                session.delete(r)
                removed += 1
    logger.info("Vacuum: removed=%d for folder=%s", removed, folder.name)
    return removed
