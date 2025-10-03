from __future__ import annotations
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text
from .db import make_engine, make_session_factory, session_scope
from .config import FolderAdapter
from .models import Base, ChatTurn

def ensure_history_schema(folder: FolderAdapter):
    engine = make_engine(folder.db)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")  # no-op if already present
    Base.metadata.create_all(engine, tables=[ChatTurn.__table__])

def load_history(session: Session, session_id: str, folder: str, max_turns: int) -> List[Dict[str, str]]:
    rows = session.execute(
        text("""
            SELECT role, content
            FROM chat_history
            WHERE session_id=:sid AND folder=:f
            ORDER BY ts DESC
            LIMIT :n
        """),
        {"sid": session_id, "f": folder, "n": max_turns * 2}
    ).mappings().all()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def save_turns(session: Session, session_id: str, folder: str, user_text: str, assistant_text: str):
    session.add_all([
        ChatTurn(session_id=session_id, folder=folder, role="user", content=user_text),
        ChatTurn(session_id=session_id, folder=folder, role="assistant", content=assistant_text),
    ])

def delete_older_than_days(session: Session, days: int):
    session.execute(text("DELETE FROM chat_history WHERE ts < now() - (:d || ' days')::interval"), {"d": days})
