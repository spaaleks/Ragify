from __future__ import annotations
import os, yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

def _resolve(v: Any) -> Any:
    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
        return os.getenv(v[2:-1], "")
    return v

# ---------- API / model ----------

@dataclass
class OpenAIConfig:
    base_url: str
    api_key: str
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OpenAIConfig":
        return OpenAIConfig(
            base_url=_resolve(d.get("base_url") or "https://api.openai.com/v1"),
            api_key=_resolve(d.get("api_key") or ""),
        )

@dataclass
class ModelConfig:
    provider: str          # "openai" | "litellm" | "gemini" | "vertex_ai"
    dim: Optional[int] = 768
    api_key: Optional[str] = None
    project: Optional[str] = None
    location: Optional[str] = None
    max_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None

# ---------- Infra ----------

@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

@dataclass
class SchedulerConfig:
    cron: Optional[str] = None
    every_minutes: Optional[int] = None

@dataclass
class PdfOcrConfig:
    url: str
    user: str
    password: str
    timeout: int

@dataclass
class S3Config:
    endpoint_url: str
    region: Optional[str]
    access_key: str
    secret_key: str
    bucket: str
    prefix: Optional[str] = None
    public_url_base: Optional[str] = None

@dataclass
class ChunkingConfig:
    size: int = 1200
    overlap: int = 150

# ---------- Adapters ----------

@dataclass
class FolderAdapter:
    name: str
    path: str
    db: DBConfig
    openai: OpenAIConfig
    embedding_model: Optional[str] = None
    chat_model: Optional[str] = None
    tag_generator: Optional[str] = None
    scheduler: Optional[SchedulerConfig] = None
    s3: Optional[S3Config] = None
    chunking: Optional[ChunkingConfig] = None
    max_retries: Optional[int] = None
    system_prompt: Optional[str] = None
    system_prompt_file: Optional[str] = None
    top_k: int = 8
    return_sources: bool = False

@dataclass
class WebhookAdapter:
    name: str
    token: str
    db: DBConfig
    openai: OpenAIConfig
    embedding_model: Optional[str] = None
    chat_model: Optional[str] = None
    s3: Optional[S3Config] = None
    chunking: Optional[ChunkingConfig] = None
    max_retries: Optional[int] = None
    system_prompt: Optional[str] = None
    system_prompt_file: Optional[str] = None
    top_k: int = 8
    return_sources: bool = False

# ---------- Root ----------

@dataclass
class AppConfig:
    global_openai: OpenAIConfig
    global_chat_model: Optional[str]
    global_embedding_model: Optional[str]
    models: Dict[str, ModelConfig]
    folders: Dict[str, FolderAdapter]
    webhooks: Dict[str, WebhookAdapter]
    scheduler: SchedulerConfig
    pdf_ocr: Optional[PdfOcrConfig] = None
    s3: Optional[S3Config] = None
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    max_retries: int = 3
    # global chat system prompt
    system_prompt: Optional[str] = None
    system_prompt_file: Optional[str] = None
    # internal: resolver for prompt files (set by loader)
    _resolve_prompt_file: Optional[Callable[[str], str]] = field(default=None, repr=False, compare=False)

# ---------- helpers ----------

def _merge_openai(base: OpenAIConfig, override: Optional[Dict[str, Any]]) -> OpenAIConfig:
    if not override:
        return base
    return OpenAIConfig(
        base_url=_resolve(override.get("base_url")) or base.base_url,
        api_key=_resolve(override.get("api_key")) or base.api_key,
    )

def _sched_from(d: Optional[Dict[str, Any]]) -> Optional[SchedulerConfig]:
    if not d:
        return None
    return SchedulerConfig(cron=d.get("cron"), every_minutes=d.get("every_minutes"))

def _s3_from(d: Optional[Dict[str, Any]]) -> Optional[S3Config]:
    if not d:
        return None
    return S3Config(
        endpoint_url=_resolve(d["endpoint_url"]),
        region=_resolve(d.get("region")),
        access_key=_resolve(d.get("access_key", "")),
        secret_key=_resolve(d.get("secret_key", "")),
        bucket=_resolve(d.get("bucket", "")),
        prefix=_resolve(d.get("prefix")),
        public_url_base=_resolve(d.get("public_url_base")),
    )

def _chunking_from(d: Optional[Dict[str, Any]]) -> Optional[ChunkingConfig]:
    if not d:
        return None
    return ChunkingConfig(
        size=int(d.get("size", 1200)),
        overlap=int(d.get("overlap", 150)),
    )

def _safe_read_under(base_dir: Path, rel_path: str) -> str:
    # Resolve rel_path against base_dir; refuse path traversal outside base_dir
    p = (base_dir / rel_path).resolve()
    base = base_dir.resolve()
    if not str(p).startswith(str(base)):
        raise ValueError(f"system_prompt_file must reside under config directory: {rel_path}")
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def _folders_from(raw_list: List[Dict[str, Any]], global_oa: OpenAIConfig, config_dir: str) -> Dict[str, FolderAdapter]:
    out: Dict[str, FolderAdapter] = {}
    base_dir = Path(config_dir)
    for item in raw_list or []:
        name = item["name"]
        if name in out:
            raise ValueError(f"duplicate folder name: {name}")
        db = DBConfig(**item["db"])
        oa = _merge_openai(global_oa, item.get("openai"))
        sched = _sched_from(item.get("scheduler"))
        s3cfg = _s3_from(item.get("s3"))
        chunking = _chunking_from(item.get("chunking"))
        # path: relative to config dir unless absolute provided
        path_str = item["path"]
        p = Path(path_str)
        if not p.is_absolute():
            p = base_dir / p
        out[name] = FolderAdapter(
            name=name,
            path=str(p),
            db=db,
            openai=oa,
            embedding_model=item.get("embedding_model"),
            chat_model=item.get("chat_model"),
            tag_generator=item.get("tag_generator"),
            scheduler=sched,
            s3=s3cfg,
            chunking=chunking,
            max_retries=int(item.get("max_retries", 0)) or None,
            system_prompt=item.get("system_prompt"),
            system_prompt_file=item.get("system_prompt_file"),
            top_k=int(item.get("top_k", 8)),
            return_sources=bool(item.get("return_sources", False)),
        )
    return out

def _webhooks_from(raw_list: List[Dict[str, Any]], global_oa: OpenAIConfig) -> Dict[str, WebhookAdapter]:
    out: Dict[str, WebhookAdapter] = {}
    for item in raw_list or []:
        name = item["name"]
        if name in out:
            raise ValueError(f"duplicate webhook name: {name}")
        db = DBConfig(**item["db"])
        oa = _merge_openai(global_oa, item.get("openai"))
        s3cfg = _s3_from(item.get("s3"))
        chunking = _chunking_from(item.get("chunking"))
        out[name] = WebhookAdapter(
            name=name,
            token=item["token"],
            db=db,
            openai=oa,
            embedding_model=item.get("embedding_model"),
            chat_model=item.get("chat_model"),
            s3=s3cfg,
            chunking=chunking,
            max_retries=int(item.get("max_retries", 0)) or None,
            system_prompt=item.get("system_prompt"),
            system_prompt_file=item.get("system_prompt_file"),
            top_k=int(item.get("top_k", 8)),
            return_sources=bool(item.get("return_sources", False)),
        )
    return out

def _models_from(raw: Dict[str, Any]) -> Dict[str, ModelConfig]:
    out: Dict[str, ModelConfig] = {}
    for name, d in (raw or {}).items():
        # dim can be absent or non-int
        dim_value = d.get("dim")
        dim = None
        if dim_value is not None:
            try:
                dim = int(dim_value)
            except (ValueError, TypeError):
                dim = 768

        def _to_int(x):
            try:
                return int(x) if x is not None else None
            except (ValueError, TypeError):
                return None

        def _to_float(x):
            try:
                return float(x) if x is not None else None
            except (ValueError, TypeError):
                return None

        out[name] = ModelConfig(
            provider=d["provider"],
            dim=dim,
            api_key=_resolve(d.get("api_key")),
            project=d.get("project"),
            location=d.get("location"),
            max_tokens=_to_int(d.get("max_tokens")),
            max_output_tokens=_to_int(d.get("max_output_tokens")),
            temperature=_to_float(d.get("temperature")),
        )
    return out

# ---------- loader ----------

def load_config(path: str) -> AppConfig:
    cfg_path = Path(path).expanduser().resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config_dir = str(cfg_path.parent)
    base_dir = Path(config_dir)

    global_oa = OpenAIConfig.from_dict(raw["global"]["openai"])
    global_chat = (raw.get("global") or {}).get("chat_model")
    global_embed = (raw.get("global") or {}).get("embedding_model")
    global_chunking = _chunking_from((raw.get("global") or {}).get("chunking")) or ChunkingConfig()
    max_retries = int((raw.get("global") or {}).get("max_retries", 3))

    models = _models_from(raw.get("models"))
    folders = _folders_from(raw.get("folders", []), global_oa, config_dir)
    webhooks = _webhooks_from(raw.get("webhooks", []), global_oa)

    sched_raw = raw.get("scheduler", {}) or {}
    scheduler = SchedulerConfig(
        cron=sched_raw.get("cron"),
        every_minutes=sched_raw.get("every_minutes"),
    )

    pdf_ocr_raw = (raw.get("global") or {}).get("pdf_ocr")
    pdf_ocr = None
    if pdf_ocr_raw:
        pdf_ocr = PdfOcrConfig(
            url=_resolve(pdf_ocr_raw["url"]),
            user=_resolve(pdf_ocr_raw["user"]),
            password=_resolve(pdf_ocr_raw["password"]),
            timeout=int(pdf_ocr_raw.get("timeout", 300)),
        )

    s3 = _s3_from((raw.get("global") or {}).get("s3"))

    appcfg = AppConfig(
        global_openai=global_oa,
        global_chat_model=global_chat,
        global_embedding_model=global_embed,
        models=models,
        folders=folders,
        webhooks=webhooks,
        scheduler=scheduler,
        pdf_ocr=pdf_ocr,
        s3=s3,
        chunking=global_chunking,
        max_retries=max_retries,
        system_prompt=(raw.get("global") or {}).get("system_prompt"),
        system_prompt_file=(raw.get("global") or {}).get("system_prompt_file"),
    )

    # Expose a safe file resolver for prompts to other modules (e.g., chat API).
    def _resolver(rel_path: str) -> str:
        if not rel_path:
            return ""
        return _safe_read_under(base_dir, rel_path)

    appcfg._resolve_prompt_file = _resolver
    return appcfg
