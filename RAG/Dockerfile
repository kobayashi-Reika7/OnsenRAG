# ============================================================
# OnsenRAG - Cloud Run 用 Dockerfile
# ============================================================
# ビルド: docker build -t onsen-rag .
# ローカル実行: docker run -p 8080:8080 --env-file .env onsen-rag
# ============================================================

FROM python:3.11-slim

# 環境変数
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# システム依存パッケージ（ビルドに必要な最小限）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 依存関係のインストール（キャッシュ活用のため先にコピー）
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# アプリケーションコードをコピー
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# HuggingFace Embeddingモデルを事前ダウンロード（起動高速化）
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-base')"

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/health')" || exit 1

# Cloud Run は PORT 環境変数でポートを指定
# gunicorn + uvicorn worker でマルチプロセス対応
CMD exec gunicorn backend.api.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --graceful-timeout 30
