from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy import text
from sqlalchemy.orm import Session
import os

# Simple semantic top-k over pgvector using cosine distance (<=>).
# Optional filters: tags_any, tags_all, mime list.

def _ensure_vector_extension(session: Session):
    """Ensure pgvector extension is installed."""
    try:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        session.commit()
    except Exception:
        # Extension might already be installed or we might not have permissions
        # In either case, we continue
        pass

def semantic_search(
    session: Session,
    qvec: list[float],
    top_k: int = 8,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    mime: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns:
      [{ "chunk_id": int, "file_id": int, "chunk_index": int, "content": str,
         "meta": dict, "mime": str, "score": float }]
    """
    # Ensure vector extension is available
    _ensure_vector_extension(session)
    
    where = []
    params: Dict[str, Any] = {"qvec": qvec, "topk": top_k}

    # metadata column name is "metadata" in DB, ORM attr is "meta"
    if tags_any:
        where.append("(metadata::jsonb ?| :tags_any)")
        params["tags_any"] = tags_any
    if tags_all:
        where.append("(SELECT array_length(:tags_all::text[], 1) = "
                     " (SELECT count(*) FROM jsonb_array_elements_text(metadata->'tags') t WHERE t.value = ANY(:tags_all)))")
        params["tags_all"] = tags_all
    if mime:
        where.append("mime = ANY(:mimes)")
        params["mimes"] = mime

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    sql = f"""
    SELECT
      c.id AS chunk_id,
      c.file_id,
      c.chunk_index,
      c.content,
      c.metadata AS meta,
      f.mime,
      f.path AS file_path,
      (1.0 - (c.embedding <=> (:qvec)::vector)) AS score
    FROM chunks c
    JOIN files f ON f.id = c.file_id
    {where_sql}
    ORDER BY c.embedding <=> (:qvec)::vector ASC
    LIMIT :topk
    """
    rows = session.execute(text(sql), params).mappings().all()
    # Add file_path to each result
    results = [dict(r) for r in rows]
    for result in results:
        if 'meta' in result and result['meta'] is not None:
            result['meta']['source'] = result.get('file_path', 'Unknown')
        else:
            result['meta'] = {'source': result.get('file_path', 'Unknown')}
    return results