# Dockerfile
FROM python:3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRPC_VERBOSITY=ERROR \
    GLOG_minloglevel=2 \
    GOOGLE_CLOUD_DISABLE_GRPC=true

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

COPY src ./src
COPY taggers ./taggers

# App listens on 8000 by default
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
