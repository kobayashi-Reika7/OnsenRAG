"""
FastAPI バックエンド - ToDo API
CORS を許可し、タスク・リストの CRUD を提供する（クラウド対応）
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import TaskCreate, TaskUpdate, TaskResponse, ListCreate, ListResponse
import store

# 環境変数（クラウドデプロイ用）
# ALLOWED_ORIGINS: カンマ区切り（例: https://myapp.vercel.app,https://myapp.netlify.app）
# PORT: サーバー待ち受けポート（デフォルト 8000）
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in _origins_env.split(",") if o.strip()]
    if _origins_env
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)

app = FastAPI(title="ToDo API", version="1.0")

# CORS: 環境変数で指定したオリジン、またはローカル開発用を許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """ヘルスチェック（ロードバランサ・コンテナオーケストレーション用）"""
    return {"status": "ok"}


# ===== タスク API =====

@app.get("/tasks", response_model=list[TaskResponse])
def get_tasks():
    """タスク一覧を返す"""
    return store.get_all_tasks()


@app.post("/tasks", response_model=TaskResponse)
def create_task(task: TaskCreate):
    """タスクを追加する"""
    data = task.model_dump()
    created = store.add_task(data)
    return created


@app.put("/tasks/{task_id}", response_model=TaskResponse)
def update_task(task_id: int, task: TaskUpdate):
    """タスクを更新する（部分更新可）"""
    data = {k: v for k, v in task.model_dump().items() if v is not None}
    updated = store.update_task(task_id, data)
    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return updated


@app.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: int):
    """タスクを削除する"""
    if not store.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return None


# ===== リスト API =====

@app.get("/lists", response_model=list[ListResponse])
def get_lists():
    """リスト一覧を返す"""
    return store.get_all_lists()


@app.post("/lists", response_model=ListResponse)
def create_list(lst: ListCreate):
    """リストを追加する"""
    created = store.add_list(lst.name)
    return created


@app.delete("/lists/{list_id}", status_code=204)
def delete_list(list_id: int):
    """リストを削除する（デフォルトは不可）"""
    if not store.delete_list(list_id):
        raise HTTPException(status_code=400, detail="Cannot delete default list or not found")
    return None
