# Day4 ToDo API - クラウドデプロイ用
# ビルド: docker build -t day4-todo-api .
# 実行: docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://your-frontend.vercel.app day4-todo-api

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py models.py store.py ./

# クラウドでは多くの場合 PORT が注入される（0.0.0.0 で待ち受ける）
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
