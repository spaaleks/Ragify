from __future__ import annotations
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import DBConfig

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
