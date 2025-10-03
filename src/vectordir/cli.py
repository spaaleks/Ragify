from __future__ import annotations
import logging, os
import argparse
from .config import load_config
from .indexer import index_folder_adapter, vacuum_deleted_files

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GOOGLE_CLOUD_DISABLE_GRPC", "true")

level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

def main():
    parser = argparse.ArgumentParser(prog="vectordir")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    idx = sub.add_parser("index", help="Index one folder or all")
    idx.add_argument("--folder", default="all", help="Folder name or 'all'")
    idx.add_argument("--force", action="store_true")
    idx.add_argument("--max-files", type=int, default=None)

    vac = sub.add_parser("vacuum", help="Vacuum one folder or all")
    vac.add_argument("--folder", default="all")

    ls = sub.add_parser("list", help="List configured folders/webhooks")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "list":
        print("folders:", ", ".join(cfg.folders.keys()) or "(none)")
        print("webhooks:", ", ".join(cfg.webhooks.keys()) or "(none)")
        return

    targets = list(cfg.folders.values()) if (
        (args.cmd in {"index", "vacuum"}) and args.folder == "all"
    ) else [cfg.folders[args.folder]]

    if args.cmd == "index":
        for f in targets:
            index_folder_adapter(f, force_reindex=args.force, max_files=args.max_files, appcfg=cfg)
    elif args.cmd == "vacuum":
        for f in targets:
            removed = vacuum_deleted_files(f, appcfg=cfg)
            print(f"{f.name}: removed={removed}")

if __name__ == "__main__":
    main()
