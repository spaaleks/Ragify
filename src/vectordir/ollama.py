from __future__ import annotations
import base64
import os
from typing import Dict, Optional

from .config import ModelConfig


def resolve_base_url(model: ModelConfig) -> str:
    return model.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def build_headers(model: ModelConfig) -> Optional[Dict[str, str]]:
    if model.username and model.password:
        token = base64.b64encode(f"{model.username}:{model.password}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}"}
    return None
