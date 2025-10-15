# src/webhook/ingest.py
from __future__ import annotations
import hashlib, os, tempfile, logging, json
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Tuple, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Depends, Query, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from langchain.schema import Document

from vectordir.config import AppConfig, WebhookAdapter, S3Config, ChunkingConfig
from vectordir.db import make_engine, make_session_factory, session_scope, ensure_match_documents_function
from vectordir.models import Base, File as DBFile, Chunk, FileStatus
from vectordir.loaders import split_docs, guess_mime, load_pdf_with_ocr
from vectordir.indexer import _embedding_client  # reuse factory from indexer

logger = logging.getLogger("vectordir.webhook")
router = APIRouter(prefix="/webhook", tags=["webhook"])
bearer = HTTPBearer(auto_error=False)

# ----------------- helpers -----------------

def _ensure_schema(engine, embedding_dim: int):
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        ensure_match_documents_function(conn, embedding_dim)

def _line_ranges_for_chunks(full_text: str, chunks: List[str]) -> List[Tuple[int, int]]:
    joined = full_text
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

def _webhook_or_401(appcfg: AppConfig, name: str, creds: Optional[HTTPAuthorizationCredentials]) -> WebhookAdapter:
    if name not in appcfg.webhooks:
        raise HTTPException(status_code=404, detail="webhook not found")
    wh = appcfg.webhooks[name]
    token = (creds.credentials if creds and creds.scheme.lower() == "bearer" else None)
    if not token or token != wh.token:
        raise HTTPException(status_code=401, detail="invalid token")
    return wh

def _resolve_embedding_dim(appcfg: AppConfig, adapter: WebhookAdapter) -> int:
    model_name = adapter.embedding_model or appcfg.global_embedding_model
    if not model_name:
        return 768
    model_spec = appcfg.models.get(model_name)
    try:
        dim = int(getattr(model_spec, "dim", 768) or 768)
    except (TypeError, ValueError):
        dim = 768
    return dim if dim > 0 else 768


def _engine_session(appcfg: AppConfig, wh: WebhookAdapter):
    dim = _resolve_embedding_dim(appcfg, wh)
    os.environ["VECTOR_EMBED_DIM"] = str(dim)
    engine = make_engine(wh.db)
    _ensure_schema(engine, dim)
    Session = make_session_factory(engine)
    return engine, Session

def _upsert_file(session: Session, vpath: str, mtime_ns: int, size_bytes: int, mime: str) -> DBFile:
    existing = session.scalar(select(DBFile).where(DBFile.path == vpath))
    if existing:
        session.execute(delete(Chunk).where(Chunk.file_id == existing.id))
        existing.mtime_ns = mtime_ns
        existing.size_bytes = size_bytes
        existing.mime = mime
        session.add(existing)
        return existing
    f = DBFile(path=vpath, mtime_ns=mtime_ns, size_bytes=size_bytes, mime=mime)
    session.add(f)
    session.flush()
    return f

def _chunk_cfg(appcfg: AppConfig, wh: WebhookAdapter) -> ChunkingConfig:
    return wh.chunking or appcfg.chunking

def _embedding_for_webhook(appcfg: AppConfig, wh: WebhookAdapter):
    model_name = wh.embedding_model or appcfg.global_embedding_model
    if not model_name:
        raise HTTPException(status_code=500, detail="no embedding model configured")
    model_spec = appcfg.models.get(model_name)
    if model_spec is None:
        raise HTTPException(status_code=500, detail=f"embedding model '{model_name}' not found in config")
    dim = getattr(model_spec, "dim", 768) or 768
    os.environ["VECTOR_EMBED_DIM"] = str(dim)
    return _embedding_client(model_name, wh, appcfg), model_name, dim

def _derive_key(explicit: str | None, filename: str | None, blob: bytes | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if blob:
        return hashlib.sha256(blob).hexdigest()
    return hashlib.sha256((filename or "").encode("utf-8")).hexdigest()

def _vpath_for(webhook: str, key: str) -> str:
    return f"webhook://{webhook}/k/{key}"

def _now_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1_000_000_000)

def _parse_tags_field(tags: Optional[str]) -> List[str]:
    if not tags:
        return []
    s = tags.strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            arr = json.loads(s)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            return []
    return [t.strip() for t in s.split(",") if t.strip()]

# ----------------- payload schema -----------------

class JsonIngest(BaseModel):
    key: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    mime: Optional[str] = None

# ----------------- endpoints -----------------

@router.post("/{name}/upload")
async def ingest_file(
    name: str,
    file: UploadFile = File(...),
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    request: Request = None,
    key: Optional[str] = Form(default=None),
    tags: Optional[str] = Form(default=None),  # JSON array or comma-separated
):
    appcfg: AppConfig = request.app.state.appcfg
    wh = _webhook_or_401(appcfg, name, creds)
    _, Session = _engine_session(appcfg, wh)

    blob = await file.read()
    mime = file.content_type or guess_mime(Path(file.filename))
    k = _derive_key(key, file.filename, blob)
    vpath = _vpath_for(name, k)
    mtime_ns = _now_ns()

    if mime == "application/pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(blob); tmp.flush()
            docs = load_pdf_with_ocr(Path(tmp.name), appcfg, wh.s3)
    else:
        text = blob.decode("utf-8", errors="ignore")
        docs = [Document(page_content=text, metadata={"source": vpath})]

    parsed_tags = _parse_tags_field(tags)
    return _persist_docs(Session, appcfg, wh, vpath, mtime_ns, len(blob), mime, docs, parsed_tags)


@router.post("/{name}/ingest")
async def ingest_json(
    name: str,
    payload: JsonIngest,
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    request: Request = None,
):
    appcfg: AppConfig = request.app.state.appcfg
    wh = _webhook_or_401(appcfg, name, creds)
    _, Session = _engine_session(appcfg, wh)

    if not payload.content:
        raise HTTPException(status_code=400, detail="content missing")
    text = payload.content
    mime = payload.mime or "text/markdown"
    k = _derive_key(payload.key, filename=None, blob=text.encode("utf-8"))
    vpath = _vpath_for(name, k)
    mtime_ns = _now_ns()
    docs = [Document(page_content=text, metadata={"source": vpath})]

    return _persist_docs(Session, appcfg, wh, vpath, mtime_ns, len(text.encode("utf-8")), mime, docs, payload.tags or [])


@router.post("/{name}/raw")
async def ingest_raw(
    name: str,
    request: Request,
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    key: Optional[str] = Query(default=None),
    tags: Optional[str] = Query(default=None),       # JSON array or comma-separated
    mime: Optional[str] = Query(default=None),
):
    appcfg: AppConfig = request.app.state.appcfg
    wh = _webhook_or_401(appcfg, name, creds)
    _, Session = _engine_session(appcfg, wh)

    blob = await request.body()
    mime = mime or "application/octet-stream"
    k = _derive_key(key, filename=None, blob=blob)
    vpath = _vpath_for(name, k)
    mtime_ns = _now_ns()

    if mime == "application/pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(blob); tmp.flush()
            docs = load_pdf_with_ocr(Path(tmp.name), appcfg, wh.s3)
    else:
        try:
            text = blob.decode("utf-8")
        except Exception:
            raise HTTPException(status_code=415, detail="unsupported non-text content; use /upload for files")
        docs = [Document(page_content=text, metadata={"source": vpath})]

    parsed_tags = _parse_tags_field(tags)
    return _persist_docs(Session, appcfg, wh, vpath, mtime_ns, len(blob), mime, docs, parsed_tags)

# -------- deletion / vacuum --------

@router.delete("/{name}/content")
def delete_content(
    name: str,
    key: Optional[str] = Query(default=None, description="Exact opaque key"),
    key_prefix: Optional[str] = Query(default=None, description="Delete all with key prefix"),
    dry_run: bool = Query(default=False),
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    request: Request = None,
):
    appcfg: AppConfig = request.app.state.appcfg
    wh = _webhook_or_401(appcfg, name, creds)
    _, Session = _engine_session(appcfg, wh)

    if not key and not key_prefix:
        raise HTTPException(status_code=400, detail="key or key_prefix required")

    base = f"webhook://{name}/k/"
    removed = 0
    with session_scope(Session) as session:
        q = select(DBFile).where(DBFile.path == base + key) if key else select(DBFile).where(DBFile.path.like(base + key_prefix + "%"))
        rows = session.scalars(q).all()
        if dry_run:
            return {"status": "dry_run", "matched": len(rows)}
        for r in rows:
            session.execute(delete(Chunk).where(Chunk.file_id == r.id))
            session.delete(r)
            removed += 1

    return {"status": "ok", "removed": removed, "mode": "key" if key else "key_prefix"}

@router.post("/{name}/vacuum-orphans")
def vacuum_orphans(
    name: str,
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    request: Request = None,
):
    appcfg: AppConfig = request.app.state.appcfg
    wh = _webhook_or_401(appcfg, name, creds)
    _, Session = _engine_session(appcfg, wh)

    removed = 0
    with session_scope(Session) as session:
        rows = session.query(DBFile).where(DBFile.path.like(f"webhook://{name}/k/%")).all()
        for r in rows:
            has_chunks = session.scalar(select(Chunk.id).where(Chunk.file_id == r.id).limit(1))
            if not has_chunks:
                session.delete(r)
                removed += 1
    return {"status": "ok", "removed": removed}

# -------------- core persist --------------

def _persist_docs(
    Session, appcfg: AppConfig, wh: WebhookAdapter,
    vpath: str, mtime_ns: int, size_bytes: int, mime: str,
    docs: List[Document], preset_tags: Optional[List[str]] = None,
):
    embeddings, model_name, dim = _embedding_for_webhook(appcfg, wh)
    chunk_cfg = wh.chunking or appcfg.chunking

    chunks_docs = split_docs(docs, chunk_size=chunk_cfg.size, chunk_overlap=chunk_cfg.overlap)
    texts = [c.page_content for c in chunks_docs]
    if not texts:
        raise HTTPException(status_code=400, detail="no text to index")

    full_text = "\n".join([d.page_content for d in docs])
    line_ranges = _line_ranges_for_chunks(full_text, texts)
    vectors = embeddings.embed_documents(texts)

    with session_scope(Session) as session:
        file_row = _upsert_file(session, vpath, mtime_ns, size_bytes, mime)
        file_row.status = FileStatus.PROCESSING
        session.add(file_row); session.commit()

        tags = preset_tags or []
        for idx, (doc, vec, (frm, to)) in enumerate(zip(chunks_docs, vectors, line_ranges)):
            meta = {"loc": {"lines": {"from": frm, "to": to}}, "tags": tags, "blobType": mime}
            session.add(Chunk(
                file_id=file_row.id,
                chunk_index=idx,
                content=doc.page_content,
                meta=meta,
                embedding=vec,
            ))

        file_row.status = FileStatus.SUCCESS
        file_row.processed_at = datetime.now(UTC)
        file_row.last_error = None
        session.add(file_row); session.commit()

    return {"status": "ok", "path": vpath, "chunks": len(texts), "model": model_name, "dim": dim, "mime": mime}
