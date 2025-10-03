# taggers/custom_tagger.py
from __future__ import annotations
from pathlib import Path
from typing import List, Pattern, Dict, Any
import re, os, json

try:
    import yaml  # optional; only needed if you use a YAML rules file
except Exception:
    yaml = None

class CustomTagger:
    """
    Pluggable tag generator.
    - Loads optional rules from TAGGER_CONFIG (path to .yml/.yaml/.json).
    - Falls back to sane defaults if no config is provided.
    - Returns a flat list[str] of tags.

    Config schema (JSON/YAML):
    {
      "rules": [
        {"pattern": "(?i)invoice", "tag": "invoice"},
        {"pattern": "(?i)contract", "tag": "contract"},
        {"pattern": "(?i)\\b20\\d{2}\\b", "extract": true, "prefix": "year:"},
        {"pattern": "(?i)/projects/([^/]+)/", "group": 1, "prefix": "project:"}
      ],
      "ext_map": {".pdf": "ext:pdf", ".md": "ext:md"},
      "dir_as_tags": true,     # add each directory name as tag
      "basename_tokens": true, # split filename by non-word and add short tokens
      "min_token_len": 3,
      "max_tokens": 8
    }
    """

    def __init__(self):
        self.cfg = self._load_cfg()
        self._compiled: List[Dict[str, Any]] = []
        for r in self.cfg.get("rules", []):
            try:
                cregex: Pattern[str] = re.compile(r["pattern"])
                self._compiled.append({
                    "regex": cregex,
                    "tag": r.get("tag"),
                    "prefix": r.get("prefix"),
                    "group": r.get("group"),
                    "extract": bool(r.get("extract", False)),
                })
            except Exception:
                # ignore bad rule
                continue
        self.ext_map: Dict[str, str] = {k.lower(): v for k, v in self.cfg.get("ext_map", {}).items()}
        self.dir_as_tags: bool = bool(self.cfg.get("dir_as_tags", True))
        self.basename_tokens: bool = bool(self.cfg.get("basename_tokens", True))
        self.min_token_len: int = int(self.cfg.get("min_token_len", 3))
        self.max_tokens: int = int(self.cfg.get("max_tokens", 8))

    # --- public API used by indexer ---
    def generate(self, path: Path, context: Dict[str, Any]) -> List[str]:
        """
        path: pathlib.Path to the file
        context: dict with at least {mtime_ns, size_bytes, mime}
        returns: list[str] tags
        """
        tags: List[str] = []
        p_str = str(path)
        mime = context.get("mime")

        # 1) extension mapping
        ext = path.suffix.lower()
        if ext in self.ext_map:
            tags.append(self.ext_map[ext])

        # 2) mime as tag
        if isinstance(mime, str) and mime:
            tags.append(f"mime:{mime}")

        # 3) regex rules
        for rule in self._compiled:
            m = rule["regex"].search(p_str)
            if not m:
                continue
            if rule["tag"]:
                tags.append(rule["tag"])
            if rule["extract"]:
                # whole match or a capture group
                if rule.get("group") is not None:
                    try:
                        val = m.group(int(rule["group"]))
                    except Exception:
                        val = m.group(0)
                else:
                    val = m.group(0)
                if val:
                    prefix = rule.get("prefix") or ""
                    tags.append(f"{prefix}{val}" if prefix else str(val))
            elif rule.get("group") is not None:
                try:
                    val = m.group(int(rule["group"]))
                    if val:
                        prefix = rule.get("prefix") or ""
                        tags.append(f"{prefix}{val}" if prefix else str(val))
                except Exception:
                    pass
            elif rule.get("prefix"):
                # prefix with no extract/group: add the prefix literal
                tags.append(str(rule["prefix"]))

        # 4) directory names as tags
        if self.dir_as_tags:
            for part in path.parent.parts:
                t = self._norm_token(part)
                if t:
                    tags.append(f"dir:{t}")

        # 5) filename tokens
        if self.basename_tokens:
            base = path.stem
            toks = [self._norm_token(t) for t in re.split(r"\W+", base)]
            toks = [t for t in toks if t and len(t) >= self.min_token_len]
            if self.max_tokens > 0:
                toks = toks[: self.max_tokens]
            tags.extend({f"name:{t}" for t in toks})

        # 6) size buckets
        try:
            size = int(context.get("size_bytes", 0))
            if size >= 10_000_000:
                tags.append("size:10mb_plus")
            elif size >= 1_000_000:
                tags.append("size:1mb_plus")
            elif size >= 100_000:
                tags.append("size:100kb_plus")
        except Exception:
            pass

        # dedupe while preserving order
        seen = set()
        out: List[str] = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    # --- helpers ---
    def _norm_token(self, s: str) -> str:
        s = s.strip().lower().replace(" ", "-")
        return re.sub(r"[^a-z0-9._-]+", "", s)

    def _load_cfg(self) -> dict:
        path = os.getenv("TAGGER_CONFIG")
        if not path:
            # default config
            return {
                "rules": [
                    {"pattern": r"(?i)\binvoice\b", "tag": "invoice"},
                    {"pattern": r"(?i)\bcontract\b", "tag": "contract"},
                    {"pattern": r"(?i)\b20\d{2}\b", "extract": True, "prefix": "year:"},
                    {"pattern": r"(?i)/projects/([^/]+)/", "group": 1, "prefix": "project:"},
                ],
                "ext_map": {".pdf": "ext:pdf", ".md": "ext:md", ".txt": "ext:txt", ".docx": "ext:docx"},
                "dir_as_tags": True,
                "basename_tokens": True,
                "min_token_len": 3,
                "max_tokens": 8,
            }
        # load from file
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.endswith((".yml", ".yaml")):
                    if yaml is None:
                        raise RuntimeError("pyyaml not installed but YAML tagger config requested")
                    return yaml.safe_load(f) or {}
                return json.load(f)
        except Exception:
            return {}
