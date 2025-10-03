# Spal.Ragify

Spal.Ragify is a modular indexing and retrieval system built around **PostgreSQL + pgvector**.  
It ingests documents from local folders or via webhooks, generates embeddings using multiple providers (OpenAI, Gemini, Vertex AI), and stores both text chunks and metadata for retrieval pipelines.

---

## Features

- **Config-driven** via `config.yml`  
- **Folder indexing** with scheduler (cron or interval)  
- **Webhook ingestion** for direct uploads, JSON, or raw streams  
- **Multiple embedding backends**: OpenAI, Gemini, Vertex AI  
- **OCR for PDFs** via external service (SPAL PDF OCR)  
- **Optional S3 storage** for page images, with automatic URL rewriting in Markdown  
- **Tagging system**: pluggable taggers for custom metadata extraction  
- **Retries and status tracking**: files marked `pending`, `processing`, `success`, `failed`, `skipped` in DB  

---

## Configuration

All settings are centralized in `config.yml`.

Example: [`config.example.yml`](config.example.yml).

## Running

### Requirements

* Python 3.14+
* PostgreSQL with `pgvector` extension
* Optional: Docker + MinIO for S3 storage
* Optional: SPAL PDF OCR service for OCR

## Running with Docker

You can deploy **Spal.Ragify** either standalone (minimal mode) or with supporting services (Postgres + MinIO).

### Minimal Compose (app only)

This runs the app container.  
You must provide your own Postgres and (optionally) S3 services.

```yaml
services:
  ragify:
    image: spaleks/ragify:latest
    container_name: ragify-app
    restart: always
    environment:
      OPENAI_API_KEY: "your-openai-key"
    volumes:
      - ./config.yml:/config/config.yml:ro
      - ./data:/data:ro
    ports:
      - "8000:8000"
```

### Full Compose (with Postgres + MinIO)

This starts **Ragify**, **Postgres (pgvector)**, and **MinIO** (S3-compatible storage).
All state is stored under `./volumes`.

```yaml
services:
  db:
    image: pgvector/pgvector:pg17
    container_name: ragify-db
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vectordb
    ports:
      - "5432:5432"
    volumes:
      - ./volumes/db:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    container_name: ragify-minio
    restart: always
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    ports:
      - "9000:9000"   # S3 API
      - "9001:9001"   # Console
    volumes:
      - ./volumes/minio:/data

  ragify:
    image: spaleks/ragify:latest
    container_name: ragify-app
    restart: always
    depends_on:
      - db
      - minio
    environment:
      OPENAI_API_KEY: "your-openai-key"
      GEMINI_API_KEY: "your-gemini-key"
      S3_ACCESS_KEY: "minio"
      S3_SECRET_KEY: "minio123"
    volumes:
      - ./config.yml:/config/config.yml:ro
      - ./data:/data:ro
      - ./volumes/ragify:/app/storage
    ports:
      - "8000:8000"
```


**Note:**

* Replace secrets (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `S3_SECRET_KEY`) with secure values or load them from an `.env` file.
* Mount your `data/` folder for document ingestion.

### Full Compose (with Postgres + MinIO + Spal OCR)

This setup runs **Ragify**, **Postgres (pgvector)**, **MinIO**, and the **Spal PDF OCR service**.  

```yaml
version: "3.9"

services:
  db:
    image: ankane/pgvector:latest
    container_name: ragify-db
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vectordb
    ports:
      - "5432:5432"
    volumes:
      - ./volumes/db:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    container_name: ragify-minio
    restart: always
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./volumes/minio:/data

  pdf_ocr:
    image: spaleks/pdf-ocr
    container_name: pdf_ocr
    restart: unless-stopped
    volumes:
      - ./.env:/app/.env:ro
    ports:
      - "31800:8000"

  ragify:
    image: spaleks/ragify:latest
    container_name: ragify-app
    restart: always
    depends_on:
      - db
      - minio
      - pdf_ocr
    environment:
      VECTOR_DIR_CONFIG_PATH: /config/config.yml
      OPENAI_API_KEY: "your-openai-key"
      GEMINI_API_KEY: "your-gemini-key"
      S3_ACCESS_KEY: "minio"
      S3_SECRET_KEY: "minio123"
    volumes:
      - ./config.yml:/config/config.yml:ro
      - ./data:/data:ro
      - ./volumes/ragify:/app/storage
    ports:
      - "8000:8000"
```

**Notes**

* `pdf_ocr` service is exposed on port `31800`.
* Configure `global.pdf_ocr.url` in `config.yml` as `http://pdf_ocr:8000/convert`.
* Provide OCR credentials via `./.env` file (mounted into `/app/.env`).

---

## Setup without Docker

```bash
# clone
git clone https://github.com/yourname/spal-ragify.git
cd spal-ragify

# create venv
python -m venv .venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt
```

### Run scheduler + API

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --app-dir src
```

* Health endpoint: `GET /health`
* Indexing runs in background according to configured schedules.

---

## Webhook Usage Examples

Replace `<WEBHOOK>` with your configured webhook name and `<TOKEN>` with the secret token.

### Upload a file
```sh
curl -X POST "http://localhost:8000/webhook/<WEBHOOK>/upload" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@./example.pdf" \
  -F "key=my-doc-key" \
  -F 'tags=["projectX","ocr"]'
```

### Ingest prepared JSON

```sh
curl -X POST "http://localhost:8000/webhook/<WEBHOOK>/ingest" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
        "key": "doc-123",
        "content": "# Title\n\nSome **markdown** text",
        "tags": ["manual","json"],
        "mime": "text/markdown"
      }'
```

### Ingest raw binary or plain text

Binary (e.g. PDF):

```sh
curl -X POST "http://localhost:8000/webhook/<WEBHOOK>/raw?mime=application/pdf&key=mydoc" \
  -H "Authorization: Bearer <TOKEN>" \
  --data-binary "@./example.pdf"
```

Plain text:

```sh
echo "Hello world text from CLI" | \
curl -X POST "http://localhost:8000/webhook/<WEBHOOK>/raw?mime=text/plain&key=cli-1&tags=quick,cli" \
  -H "Authorization: Bearer <TOKEN>" \
  --data-binary @-
```

### Delete by key or prefix

Delete exact key:

```sh
curl -X DELETE "http://localhost:8000/webhook/<WEBHOOK>/content?key=my-doc-key" \
  -H "Authorization: Bearer <TOKEN>"
```

Delete by prefix:

```sh
curl -X DELETE "http://localhost:8000/webhook/<WEBHOOK>/content?key_prefix=myproj-" \
  -H "Authorization: Bearer <TOKEN>"
```

Dry-run mode:

```sh
curl -X DELETE "http://localhost:8000/webhook/<WEBHOOK>/content?key_prefix=myproj-&dry_run=true" \
  -H "Authorization: Bearer <TOKEN>"
```

### Vacuum orphaned entries

```sh
curl -X POST "http://localhost:8000/webhook/<WEBHOOK>/vacuum-orphans" \
  -H "Authorization: Bearer <TOKEN>"
```

---

## Tagger Example

A tagger is a Python class implementing:

```python
class CustomTagger:
    def generate(self, path: Path, context: dict) -> List[str]:
        return ["invoice", "year:2025"]
```

See [`taggers/custom_tagger.py`](taggers/custom_tagger.py).

---

## Roadmap

* [x] Folder indexing with retries
* [x] OCR PDF support + S3 image storage
* [x] Tagging system
* [x] Webhook ingestion
* [x] Query API for search and RAG pipelines
* [ ] MCP Server(s)

---

## License

MIT License
