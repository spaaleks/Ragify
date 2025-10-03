from __future__ import annotations
import os, logging, time, json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

from vectordir.config import AppConfig, FolderAdapter
from vectordir.db import make_engine, make_session_factory, session_scope
from vectordir.indexer import _embedding_client
from chatapi.retrieval import semantic_search
from vectordir.history import ensure_history_schema, load_history, save_turns

router = APIRouter(prefix="/v1", tags=["chat"])
plain_router = APIRouter(tags=["chat"])
log = logging.getLogger("vectordir.chatapi")

# ---------- schemas ----------
class ChatMessage(BaseModel):
    role: str
    content: str

class RagOptions(BaseModel):
    folder: Optional[str] = None
    webhook: Optional[str] = None
    top_k: int = 8
    filters: Dict[str, Any] = Field(default_factory=dict)
    return_sources: bool = False

class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stream: Optional[bool] = False
    rag: Optional[RagOptions] = None
    session_id: Optional[str] = None

# ---------- alias helpers ----------
def _list_model_aliases(appcfg: AppConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, f in appcfg.folders.items():
        out.append({
            "id": f"rag:folder:{name}",
            "object": "model",
            "owned_by": "ragify",
            "rag_scope": {"folder": name},
            "streaming": True,
        })
    for name, _w in appcfg.webhooks.items():
        out.append({
            "id": f"rag:webhook:{name}",
            "object": "model",
            "owned_by": "ragify",
            "rag_scope": {"webhook": name},
            "streaming": True,
        })
    return out

def _resolve_alias(model_name: str, appcfg: AppConfig):
    # returns (folder_adapter_or_None, resolved_model_name)
    if model_name.startswith("rag:folder:"):
        fname = model_name.split(":", 2)[2]
        f = appcfg.folders.get(fname)
        if not f:
            raise HTTPException(404, f"folder '{fname}' not found")
        resolved = f.chat_model or appcfg.global_chat_model
        if not resolved:
            raise HTTPException(400, "no chat model configured for alias")
        return f, resolved
    if model_name.startswith("rag:webhook:"):
        wname = model_name.split(":", 2)[2]
        w = appcfg.webhooks.get(wname)
        if not w:
            raise HTTPException(404, f"webhook '{wname}' not found")
        resolved = w.chat_model or appcfg.global_chat_model
        if not resolved:
            raise HTTPException(400, "no chat model configured for alias")
        folder = next(iter(appcfg.folders.values())) if appcfg.folders else None
        return folder, resolved
    return None, model_name

# ---------- SSE helpers (OpenAI-compatible) ----------
def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

def _sse_done() -> str:
    return "data: [DONE]\n\n"

async def _ping(req_id: str):
    return f": ping {int(time.time())}\n\n"

# ---------- provider factory ----------
def _chat_client(model_name: str, folder: Optional[FolderAdapter], appcfg: AppConfig, *, streaming: bool=False):
    m = appcfg.models.get(model_name) or (_ for _ in ()).throw(HTTPException(400, f"unknown model: {model_name}"))
    if m.provider in ("openai", "litellm"):
        return ChatOpenAI(
            model=model_name,
            api_key=(folder.openai.api_key if folder else appcfg.global_openai.api_key) or os.getenv("OPENAI_API_KEY", ""),
            base_url=(folder.openai.base_url if folder else appcfg.global_openai.base_url) or "https://api.openai.com/v1",
            temperature=0.0,
            streaming=streaming,
        )
    if m.provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=m.api_key or os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
            streaming=streaming,
            convert_system_message_to_human=True
        )
    if m.provider == "vertex_ai":
        return ChatVertexAI(
            model_name=model_name,
            project=m.project,
            location=m.location,
            temperature=0.0,
            streaming=streaming,
        )
    raise HTTPException(400, f"unsupported chat provider: {m.provider}")

# ---------- prompt helpers ----------
def _resolve_system_prompt(folder: Optional[FolderAdapter], appcfg: AppConfig) -> Optional[str]:
    if folder and folder.system_prompt:
        return folder.system_prompt
    if folder and folder.system_prompt_file and appcfg._resolve_prompt_file:
        return appcfg._resolve_prompt_file(folder.system_prompt_file)
    if appcfg.system_prompt:
        return appcfg.system_prompt
    if appcfg.system_prompt_file and appcfg._resolve_prompt_file:
        return appcfg._resolve_prompt_file(appcfg.system_prompt_file)
    return None

def _build_prompt(messages: List[ChatMessage], system_prompt: Optional[str], ctx_snippets: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    if ctx_snippets:
        out.append({"role": "system", "content": "Use the following context:\n\n" + "\n\n---\n\n".join(ctx_snippets[:20])})
    for m in messages:
        if m.role in ("system", "user", "assistant"):
            out.append({"role": m.role, "content": m.content})
    return out

# ---------- routes ----------
@router.get("/models")
def list_models(request: Request):
    appcfg = getattr(request.app.state, "appcfg", None)
    if not appcfg:
        raise HTTPException(status_code=503, detail="config not loaded")
    return {"object": "list", "data": _list_model_aliases(appcfg)}

@plain_router.get("/models")
def list_models_plain(request: Request):
    appcfg = getattr(request.app.state, "appcfg", None)
    if not appcfg:
        raise HTTPException(status_code=503, detail="config not loaded")
    return {"object": "list", "data": _list_model_aliases(appcfg)}

@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionsRequest, request: Request):
    appcfg: AppConfig = request.app.state.appcfg
    if not appcfg:
        raise HTTPException(503, "config not loaded")

    # ---------- resolve model + alias scope ----------
    folder_scope: Optional[FolderAdapter] = None
    model_name_input = req.model or appcfg.global_chat_model
    if not model_name_input:
        raise HTTPException(400, "no chat model provided")
    folder_scope, resolved_model = _resolve_alias(model_name_input, appcfg)

    prov = appcfg.models[resolved_model].provider
    cfg_model = appcfg.models.get(resolved_model)

    def _normalize_limit(val: Optional[int]) -> Optional[int]:
        if val is None:
            return None
        try:
            iv = int(val)
            return None if iv < 0 else iv  # -1 => no explicit cap
        except (ValueError, TypeError):
            return None

    def _gen_kwargs() -> Dict[str, Any]:
        t = req.temperature if req.temperature is not None else (getattr(cfg_model, "temperature", None) or 0.0)

        def _norm(x: Optional[int]) -> Optional[int]:
            try:
                return None if x is None or int(x) < 0 else int(x)
            except (ValueError, TypeError):
                return None

        req_lim = _norm(req.max_tokens) if req.max_tokens is not None else None
        cfg_lim = _norm(getattr(cfg_model, "max_tokens", None)) or _norm(getattr(cfg_model, "max_output_tokens", None))

        lim = req_lim if req_lim is not None else (cfg_lim if cfg_lim is not None else 800)

        kw: Dict[str, Any] = {"temperature": t}
        if lim is not None:
            if prov in ("gemini", "vertex_ai"):
                kw["max_output_tokens"] = lim
            else:  # openai/litellm/proxies
                kw["max_tokens"] = lim
        log.debug("gen_kwargs for %s prov=%s -> %s", resolved_model, prov, kw)
        return kw

    gen_kwargs = _gen_kwargs()

    # ---------- RAG scope ----------
    auto_rag_enabled = model_name_input.startswith("rag:folder:") or model_name_input.startswith("rag:webhook:")
    rag_folder: Optional[FolderAdapter] = None
    if auto_rag_enabled:
        if model_name_input.startswith("rag:folder:"):
            rag_folder_name = model_name_input.split(":", 2)[2]
            rag_folder = appcfg.folders.get(rag_folder_name) or (_ for _ in ()).throw(HTTPException(404, f"folder '{rag_folder_name}' not found"))
        else:
            rag_folder = folder_scope
    elif req.rag and req.rag.folder:
        rag_folder = appcfg.folders.get(req.rag.folder) or (_ for _ in ()).throw(HTTPException(404, f"folder '{req.rag.folder}' not found"))
    if rag_folder is None:
        rag_folder = folder_scope

    chat = _chat_client(resolved_model, folder_scope, appcfg, streaming=bool(req.stream))

    # ---------- helpers ----------
    def _embedding_preflight(folder_ctx: FolderAdapter, appcfg: AppConfig) -> str:
        embed_model = folder_ctx.embedding_model or appcfg.global_embedding_model
        if not embed_model:
            raise HTTPException(503, "No embedding_model configured. Set global.embedding_model or folders[].embedding_model.")
        m = appcfg.models.get(embed_model)
        if not m:
            raise HTTPException(400, f"Embedding model '{embed_model}' not in registry.")
        if m.provider == "openai":
            if not (folder_ctx.openai and folder_ctx.openai.api_key):
                raise HTTPException(503, "Missing OpenAI api_key.")
        elif m.provider == "gemini":
            if not (m.api_key or os.getenv("GEMINI_API_KEY")):
                raise HTTPException(503, "Missing Gemini API key.")
        elif m.provider == "vertex_ai":
            if not (m.project and m.location):
                raise HTTPException(503, "Missing Vertex AI project/location.")
        else:
            raise HTTPException(400, f"Unsupported embedding provider: {m.provider}")
        return embed_model

    def _require_folder_context() -> FolderAdapter:
        if rag_folder:
            return rag_folder
        if folder_scope:
            return folder_scope
        raise HTTPException(503, "No folder context for RAG. GET /v1/models and select a rag:folder:* alias, or disable RAG.")

    # ---------- history config ----------
    session_id = getattr(req, "session_id", None)
    HISTORY_MAX_TURNS = 20

    # ---------- non-streaming path ----------
    if not req.stream:
        # RAG before call (blocking is fine for non-stream)
        ctx_chunks: List[Dict[str, Any]] = []
        ctx_snippets: List[str] = []
        if auto_rag_enabled or bool(req.rag):
            try:
                filters = (req.rag.filters if req.rag else {}) or {}
                tags_any = filters.get("tags_any") or filters.get("tagsAny")
                tags_all = filters.get("tags_all") or filters.get("tagsAll")
                mime = filters.get("mime") or filters.get("mimes")
                if isinstance(mime, str):
                    mime = [mime]
                user_msgs = [m.content for m in req.messages if m.role == "user"]
                query_text = user_msgs[-1] if user_msgs else ""
                folder_ctx = _require_folder_context()
                embed_model = _embedding_preflight(folder_ctx, appcfg)
                embedder = _embedding_client(embed_model, folder_ctx, appcfg)
                qvec = embedder.embed_query(query_text)
                engine = make_engine(folder_ctx.db)
                SessionFactory = make_session_factory(engine)
                with session_scope(SessionFactory) as sess:
                    ctx_chunks = semantic_search(
                        sess, qvec,
                        top_k=(req.rag.top_k if req.rag else 8) or 8,
                        tags_any=tags_any, tags_all=tags_all, mime=mime,
                    )
                ctx_snippets = [c.get("content", "") for c in ctx_chunks if c.get("content")]
            except Exception as e:
                log.warning("RAG init failed: %s", e)

        # history load
        history_msgs: List[Dict[str, str]] = []
        if session_id and rag_folder:
            try:
                ensure_history_schema(rag_folder)
                engine = make_engine(rag_folder.db)
                SessionFactory = make_session_factory(engine)
                with session_scope(SessionFactory) as s:
                    history_msgs = load_history(s, session_id, rag_folder.name, HISTORY_MAX_TURNS)
            except Exception as e:
                log.warning("history load failed: %s", e)

        # assemble prompt
        system_prompt = _resolve_system_prompt(rag_folder, appcfg)
        sys_msgs: List[Dict[str, str]] = []
        if system_prompt:
            sys_msgs.append({"role": "system", "content": system_prompt})
        if ctx_snippets:
            sys_msgs.append({"role": "system", "content": "Use the following context:\n\n" + "\n\n---\n\n".join(ctx_snippets[:20])})
        current_msgs = [{"role": m.role, "content": m.content} for m in req.messages if m.role in ("system", "user", "assistant")]
        current_msgs = [m for m in current_msgs if m["role"] != "system"]
        messages: List[Dict[str, str]] = sys_msgs + history_msgs + current_msgs
        last_user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        # call model
        resp = await chat.ainvoke(messages, **gen_kwargs)
        content = getattr(resp, "content", None)
        if not isinstance(content, str):
            try:
                content = "".join([p.get("text", "") for p in content if isinstance(p, dict)])
            except Exception:
                content = str(content)

        # persist history
        if session_id and rag_folder:
            try:
                engine = make_engine(rag_folder.db)
                SessionFactory = make_session_factory(engine)
                with session_scope(SessionFactory) as s:
                    save_turns(s, session_id, rag_folder.name, last_user_text, content or "")
            except Exception as e:
                log.warning("history save failed: %s", e)

        out = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resolved_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content or ""},
                "finish_reason": "stop",
            }],
        }
        if (req.rag and req.rag.return_sources) and ctx_chunks:
            out["sources"] = [{
                "source": ch.get("meta", {}).get("source", "Unknown"),
                "chunk_id": ch.get("chunk_id"),
                "chunk_index": ch.get("chunk_index"),
                "loc": ch.get("meta", {}).get("loc", {}).get("lines", {}),
            } for ch in ctx_chunks]
        return out

    # ---------- streaming path (primer first, RAG inside) ----------
    from starlette.concurrency import run_in_threadpool

    async def _compute_rag() -> Dict[str, Any]:
        """Run potentially blocking RAG work in a thread. Return dict with ctx_chunks/snippets and optional info."""
        result: Dict[str, Any] = {"ctx_chunks": [], "ctx_snippets": [], "info": None}
        if not (auto_rag_enabled or bool(req.rag)):
            return result

        def _work():
            try:
                filters = (req.rag.filters if req.rag else {}) or {}
                tags_any = filters.get("tags_any") or filters.get("tagsAny")
                tags_all = filters.get("tags_all") or filters.get("tagsAll")
                mime = filters.get("mime") or filters.get("mimes")
                if isinstance(mime, str):
                    mime_list = [mime]
                else:
                    mime_list = mime
                user_msgs = [m.content for m in req.messages if m.role == "user"]
                query_text = user_msgs[-1] if user_msgs else ""
                folder_ctx = _require_folder_context()
                embed_model = _embedding_preflight(folder_ctx, appcfg)
                embedder = _embedding_client(embed_model, folder_ctx, appcfg)
                qvec = embedder.embed_query(query_text)
                engine = make_engine(folder_ctx.db)
                SessionFactory = make_session_factory(engine)
                with session_scope(SessionFactory) as sess:
                    chunks = semantic_search(
                        sess, qvec,
                        top_k=(req.rag.top_k if req.rag else 8) or 8,
                        tags_any=tags_any, tags_all=tags_all, mime=mime_list,
                    )
                snippets = [c.get("content", "") for c in chunks if c.get("content")]
                return {"ctx_chunks": chunks, "ctx_snippets": snippets, "info": None}
            except Exception as e:
                return {"ctx_chunks": [], "ctx_snippets": [], "info": f"[info] RAG initialization failed; continuing without context. Reason: {e}"}

        return await run_in_threadpool(_work)

    async def _load_history_msgs() -> List[Dict[str, str]]:
        if not (session_id and rag_folder):
            return []
        def _work():
            try:
                ensure_history_schema(rag_folder)
                engine = make_engine(rag_folder.db)
                SessionFactory = make_session_factory(engine)
                with session_scope(SessionFactory) as s:
                    return load_history(s, session_id, rag_folder.name, HISTORY_MAX_TURNS)
            except Exception as e:
                log.warning("history load failed: %s", e)
                return []
        return await run_in_threadpool(_work)

    async def _save_history(user_text: str, assistant_text: str):
        if not (session_id and rag_folder):
            return
        def _work():
            try:
                engine = make_engine(rag_folder.db)
                SessionFactory = make_session_factory(engine)
                with session_scope(SessionFactory) as s:
                    save_turns(s, session_id, rag_folder.name, user_text, assistant_text)
            except Exception as e:
                log.warning("history save failed: %s", e)
        await run_in_threadpool(_work)

    async def _stream_main():
        started = int(time.time())
        req_id = f"chatcmpl-{started}"
        log.info("SSE start model=%s alias=%s", resolved_model, model_name_input)

        # primer so client renders immediately
        yield _sse({
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": started,
            "model": resolved_model,
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
        })

        yield await _ping(req_id)

        # compute RAG + history without blocking loop
        rag_result = await _compute_rag()
        if rag_result.get("info"):
            yield _sse({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": resolved_model,
                "choices": [{"index": 0, "delta": {"content": "\n" + rag_result["info"]}, "finish_reason": None}],
            })

        history_msgs = await _load_history_msgs()

        # assemble prompt now
        system_prompt = _resolve_system_prompt(rag_folder, appcfg)
        sys_msgs: List[Dict[str, str]] = []
        if system_prompt:
            sys_msgs.append({"role": "system", "content": system_prompt})
        if rag_result["ctx_snippets"]:
            sys_msgs.append({
                "role": "system",
                "content": "Use the following context:\n\n" + "\n\n---\n\n".join(rag_result["ctx_snippets"][:20])
            })

        current_msgs = [
            {"role": m.role, "content": m.content}
            for m in req.messages
            if m.role in ("system", "user", "assistant")
        ]
        current_msgs = [m for m in current_msgs if m["role"] != "system"]

        messages: List[Dict[str, str]] = sys_msgs + history_msgs + current_msgs
        last_user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        # send role chunk per OpenAI behavior
        yield _sse({
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": resolved_model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })

        # --- stream tokens (ONE try/except around both branches) ---
        tokens = 0
        out_buf: List[str] = []
        try:
            if prov in ("gemini", "vertex_ai"):
                async for ch in chat.astream(messages, **gen_kwargs):
                    token = getattr(ch, "content", None) or getattr(ch, "delta", "")
                    if token:
                        tokens += 1
                        out_buf.append(token)
                        yield _sse({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": resolved_model,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                        })
            else:
                async for ch in chat.astream(messages, **gen_kwargs):
                    token = getattr(ch, "content", None) or getattr(ch, "delta", "")
                    if token:
                        tokens += 1
                        out_buf.append(token)
                        yield _sse({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": resolved_model,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                        })

            # optional sources
            if (req.rag and req.rag.return_sources) and rag_result["ctx_chunks"]:
                lines = []
                for i, ch in enumerate(rag_result["ctx_chunks"], 1):
                    loc = ch.get("meta", {}).get("loc", {}).get("lines", {})
                    span = f"L{loc.get('from','?')}-L{loc.get('to','?')}"
                    file_path = ch.get("meta", {}).get("source", "Unknown")
                    lines.append(f"{i}. {file_path} (chunk={ch.get('chunk_id','?')}:{ch.get('chunk_index','?')} {span})")
                src_blob = "\n\n---\nSources:\n" + "\n".join(lines)
                yield _sse({
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": resolved_model,
                    "choices": [{"index": 0, "delta": {"content": src_blob}, "finish_reason": None}],
                })

            # final + done
            yield _sse({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": resolved_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            })
            log.info("SSE done model=%s tokens=%d", resolved_model, tokens)

            # persist history after streaming completes
            await _save_history(last_user_text, "".join(out_buf))

        except Exception as e:
            log.exception("SSE fatal: %s", e)
            yield _sse({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": resolved_model,
                "choices": [{"index": 0, "delta": {"content": "\n[error] " + str(e)}, "finish_reason": None}],
            })

    return StreamingResponse(
        _stream_main(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
