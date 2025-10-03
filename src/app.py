# src/app.py
from __future__ import annotations
import asyncio, logging, os
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from croniter import croniter

from vectordir.config import load_config, AppConfig, SchedulerConfig, FolderAdapter
from vectordir.indexer import index_folder_adapter, vacuum_deleted_files
from webhook.ingest import router as webhook_router
from chatapi.router import router as chat_router, plain_router as chat_plain_router
from chatapi.stubs import stub as chat_stubs

# Silence gRPC noise
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GOOGLE_CLOUD_DISABLE_GRPC", "true")

level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("vectordir.app")

CONFIG_PATH_ENV = "VECTOR_DIR_CONFIG_PATH"
DEFAULT_CFG = "config.yml"
VACUUM_EVERY_MINUTES = int(os.getenv("VACUUM_EVERY_MINUTES", "1440"))

# ---------- app + routers ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg_path = os.getenv(CONFIG_PATH_ENV, DEFAULT_CFG)
    # initial config load so routers have appcfg immediately
    try:
        cfg = load_config(cfg_path)
        app.state.appcfg = cfg
        log.info("Initial config loaded from %s. Folders: %s", cfg_path, ", ".join(cfg.folders.keys()) or "(none)")
    except Exception as e:
        log.error("Initial config load failed from %s: %s", cfg_path, e)
        app.state.appcfg = None

    # start supervisor (reload + schedulers)
    app.state.supervisor_task = asyncio.create_task(_supervisor(app))

    try:
        yield
    finally:
        # shutdown
        task: asyncio.Task = app.state.supervisor_task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

app = FastAPI(title="vectordir", version="0.4.1", lifespan=lifespan)
app.include_router(webhook_router)
app.include_router(chat_router)
app.include_router(chat_plain_router)  # /models for OpenWebUI
app.include_router(chat_stubs)         # /api/tags, /api/ps stubs

_last_vacuum: Dict[str, datetime] = {}

# ---------- helpers ----------
def _resolve_scheduler(folder: FolderAdapter, global_sched: SchedulerConfig) -> SchedulerConfig:
    return folder.scheduler if folder.scheduler else global_sched

def _next_sleep_seconds(sched: SchedulerConfig) -> float:
    now = datetime.now(timezone.utc)
    if sched.cron:
        it = croniter(sched.cron, now)
        nxt = it.get_next(datetime)
        return max(1.0, (nxt - now).total_seconds())
    if sched.every_minutes and sched.every_minutes > 0:
        return float(sched.every_minutes * 60)
    return 900.0

def _should_vacuum(folder_name: str) -> bool:
    last = _last_vacuum.get(folder_name)
    if last is None:
        return True
    return (datetime.now(timezone.utc) - last) >= timedelta(minutes=VACUUM_EVERY_MINUTES)

def _embed_dim(appcfg: AppConfig, model_name: Optional[str], default: int = 768) -> int:
    if not model_name:
        return default
    spec = appcfg.models.get(model_name)
    if spec is None:
        return default
    return int(getattr(spec, "dim", default) or default)

# ---------- background loops ----------
async def _folder_scheduler_loop(folder: FolderAdapter, global_sched: SchedulerConfig, appcfg: AppConfig):
    sched = _resolve_scheduler(folder, global_sched)
    fname = folder.name
    while True:
        try:
            model_name = folder.embedding_model or appcfg.global_embedding_model
            os.environ["VECTOR_EMBED_DIM"] = str(_embed_dim(appcfg, model_name, 768))

            index_folder_adapter(folder, appcfg, force_reindex=False, max_files=None)

            if _should_vacuum(fname):
                removed = vacuum_deleted_files(folder, appcfg)
                _last_vacuum[fname] = datetime.now(timezone.utc)
                log.info("Vacuum run for %s removed=%d", fname, removed)

        except Exception as e:
            log.exception("index/vacuum error in %s: %s", fname, e)

        await asyncio.sleep(_next_sleep_seconds(sched))

async def _supervisor(app: FastAPI):
    """Reload config if needed and (re)spawn folder tasks."""
    cfg_path = os.getenv(CONFIG_PATH_ENV, DEFAULT_CFG)
    tasks: List[asyncio.Task] = []
    last_loaded_sig: Optional[str] = None

    while True:
        try:
            cfg = load_config(cfg_path)
            # create a simple signature: number of folders + model keys; adjust if hot-reload detection is needed
            sig = f"{len(cfg.folders)}:{','.join(sorted(cfg.models.keys()))}"
            if sig != last_loaded_sig or getattr(app.state, "appcfg", None) is None:
                # cancel old tasks
                for t in tasks:
                    t.cancel()
                for t in tasks:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                tasks.clear()
                _last_vacuum.clear()

                app.state.appcfg = cfg
                for folder in cfg.folders.values():
                    _last_vacuum[folder.name] = datetime.min.replace(tzinfo=timezone.utc)
                    t = asyncio.create_task(_folder_scheduler_loop(folder, cfg.scheduler, cfg))
                    tasks.append(t)
                last_loaded_sig = sig
                log.info("Config (re)loaded from %s. Folders: %s", cfg_path, ", ".join(cfg.folders.keys()) or "(none)")
        except Exception as e:
            log.error("Config reload failed from %s: %s", cfg_path, e)

        await asyncio.sleep(5)

# ---------- endpoints ----------
@app.get("/health")
def health():
    cfg: Optional[AppConfig] = getattr(app.state, "appcfg", None)
    return {
        "status": "ok" if cfg else "not_ready",
        "time": datetime.now(timezone.utc).isoformat(),
        "config_loaded": cfg is not None,
        "folders": list(cfg.folders.keys()) if cfg else [],
        "vacuum_every_minutes": VACUUM_EVERY_MINUTES,
        "last_vacuum": {k: v.isoformat() for k, v in _last_vacuum.items()},
    }

@app.get("/ready")
def ready():
    return {"ready": bool(getattr(app.state, "appcfg", None))}

@app.post("/admin/vacuum")
def admin_vacuum(folder: str = "all"):
    cfg: Optional[AppConfig] = getattr(app.state, "appcfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="config not loaded")
    if folder == "all":
        total = 0
        for f in cfg.folders.values():
            total += vacuum_deleted_files(f, cfg)
            _last_vacuum[f.name] = datetime.now(timezone.utc)
        return {"status": "ok", "folders": "all", "removed": total}
    if folder not in cfg.folders:
        raise HTTPException(status_code=404, detail=f"folder '{folder}' not found")
    removed = vacuum_deleted_files(cfg.folders[folder], cfg)
    _last_vacuum[folder] = datetime.now(timezone.utc)
    return {"status": "ok", "folder": folder, "removed": removed}

@app.post("/admin/index")
async def admin_index(
    folder: str = Query(default="all"),
    force_reindex: bool = Query(default=False),
    max_files: int | None = Query(default=None, ge=1),
    async_run: bool = Query(default=False),
):
    cfg: Optional[AppConfig] = getattr(app.state, "appcfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="config not loaded")

    def _run_one(f: FolderAdapter):
        model_name = f.embedding_model or cfg.global_embedding_model
        os.environ["VECTOR_EMBED_DIM"] = str(_embed_dim(cfg, model_name, 768))
        index_folder_adapter(f, cfg, force_reindex=force_reindex, max_files=max_files)

    if folder == "all":
        if async_run:
            for f in cfg.folders.values():
                asyncio.create_task(asyncio.to_thread(_run_one, f))
            return {"status": "started", "folders": "all", "async": True}
        else:
            for f in cfg.folders.values():
                await asyncio.to_thread(_run_one, f)
            return {"status": "ok", "folders": "all", "async": False}

    if folder not in cfg.folders:
        raise HTTPException(status_code=404, detail=f"folder '{folder}' not found")

    f = cfg.folders[folder]
    if async_run:
        asyncio.create_task(asyncio.to_thread(_run_one, f))
        return {"status": "started", "folder": folder, "async": True}
    else:
        await asyncio.to_thread(_run_one, f)
        return {"status": "ok", "folder": folder, "async": False}
