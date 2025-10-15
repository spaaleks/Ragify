from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional
import mimetypes
import base64
import logging
import re

import requests
from requests.exceptions import Timeout, ConnectionError, RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import AppConfig, S3Config, ChunkingConfig

logger = logging.getLogger("vectordir")

TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv"}
DOC_EXT = {".docx"}

def guess_mime(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"

# ---------------- S3 helpers ----------------

def _s3_client(s3cfg: S3Config):
    session = boto3.session.Session()
    cfg = BotoConfig(
        region_name=s3cfg.region or "us-east-1",
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=5,
        read_timeout=60,
    )
    return session.client(
        "s3",
        endpoint_url=s3cfg.endpoint_url,
        aws_access_key_id=s3cfg.access_key,
        aws_secret_access_key=s3cfg.secret_key,
        config=cfg,
    )

def _s3_put_if_needed(client, bucket: str, key: str, data: bytes, content_type: str) -> None:
    try:
        client.head_object(Bucket=bucket, Key=key)
        logger.debug("S3: object already exists s3://%s/%s", bucket, key)
        return
    except ClientError as e:
        code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code != 404:
            raise
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    logger.info("S3: uploaded s3://%s/%s (%d bytes)", bucket, key, len(data))

def _s3_object_url(s3cfg: S3Config, key: str) -> str:
    if s3cfg.public_url_base:
        return f"{s3cfg.public_url_base.rstrip('/')}/{s3cfg.bucket}/{key}"
    if s3cfg.endpoint_url.startswith("http"):
        return f"{s3cfg.endpoint_url.rstrip('/')}/{s3cfg.bucket}/{key}"
    return f"s3://{s3cfg.bucket}/{key}"

# --------------- PDF OCR integration ---------------

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Timeout, ConnectionError)),
)
def _call_pdf_ocr(path: Path, appcfg: AppConfig) -> dict:
    if not appcfg.pdf_ocr:
        raise RuntimeError("PDF OCR service not configured in YAML (global.pdf_ocr)")
    logger.info("OCR: POST %s file=%s", appcfg.pdf_ocr.url, path.name)
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "application/pdf")}
        resp = requests.post(
            appcfg.pdf_ocr.url,
            files=files,
            auth=(appcfg.pdf_ocr.user, appcfg.pdf_ocr.password),
            timeout=(getattr(appcfg.pdf_ocr, "connect_timeout", 10) or 10, appcfg.pdf_ocr.timeout),
        )
    logger.info("OCR: %s -> %s", path.name, resp.status_code)
    resp.raise_for_status()
    return resp.json()

def _maybe_upload_ocr_images(pages: List[dict], s3cfg: Optional[S3Config]) -> Optional[dict]:
    if not s3cfg:
        logger.debug("S3: no config; skip uploads")
        return None
    client = _s3_client(s3cfg)
    prefix = (s3cfg.prefix or "").lstrip("/")
    uploaded = 0
    page_url_map: dict[int, str] = {}
    for page in pages:
        b64 = page.get("image_base64")
        ext = page.get("ext") or ".png"
        checksum = page.get("image_base64_checksum")
        page_idx = int(page.get("page", 0))
        if not (b64 and checksum):
            continue
        try:
            blob = base64.b64decode(b64)
        except Exception:
            continue
        key = f"{prefix}{checksum}{ext}"
        _s3_put_if_needed(client, s3cfg.bucket, key, blob, "image/png")
        url = _s3_object_url(s3cfg, key)
        page_url_map[page_idx] = url
        uploaded += 1
    logger.info("S3: pages processed=%d", uploaded)
    return page_url_map

def _rewrite_markdown_images(md: str, pages: List[dict], s3cfg: Optional[S3Config], page_url_map: Optional[dict]) -> str:
    if not md or not s3cfg or not page_url_map:
        return md
    pattern = re.compile(r"\((?:\.?/)?page-(\d+)\.png\)")
    def repl(m: re.Match) -> str:
        try:
            idx = int(m.group(1))
        except Exception:
            return m.group(0)
        url = page_url_map.get(idx)
        return f"({url})" if url else m.group(0)
    return pattern.sub(repl, md)

def load_pdf_with_ocr(path: Path, appcfg: AppConfig, s3cfg: Optional[S3Config]) -> List[Document]:
    try:
        data = _call_pdf_ocr(path, appcfg)
    except RequestException as exc:
        logger.warning("OCR: request failed for %s (%s); falling back to local PDF loader", path.name, exc)
        return UnstructuredFileLoader(str(path)).load()
    pages = data.get("pages", [])
    page_url_map = _maybe_upload_ocr_images(pages, s3cfg)
    md_text = data.get("markdown", "")
    md_text = _rewrite_markdown_images(md_text, pages, s3cfg, page_url_map)
    logger.info("OCR: pages=%d markdown_len=%d for %s", len(pages), len(md_text), path.name)
    return [Document(page_content=md_text, metadata={"source": str(path)})]

# --------------- Generic loaders ---------------

def load_file(path: Path, appcfg: AppConfig, s3cfg: Optional[S3Config] = None) -> List[Document]:
    mime = guess_mime(path)
    if mime == "application/pdf" or path.suffix.lower() == ".pdf":
        return load_pdf_with_ocr(path, appcfg, s3cfg)
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return TextLoader(str(path), encoding="utf-8").load()
    if ext in DOC_EXT:
        return Docx2txtLoader(str(path)).load()
    return UnstructuredFileLoader(str(path)).load()

def discover_paths(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            yield p

def split_docs(docs: List[Document], chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
