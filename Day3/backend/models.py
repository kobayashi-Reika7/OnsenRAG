"""
Pydantic モデル（リクエスト・レスポンスの型定義）
Task と List のデータ構造を定義する
"""
from pydantic import BaseModel
from typing import Optional


class TaskBase(BaseModel):
    title: str
    list_id: int = 1
    is_completed: bool = False
    is_favorite: bool = False
    due_date: Optional[str] = None
    memo: str = ""
    time: int = 0  # タイマー経過秒数（保存用）


class TaskCreate(TaskBase):
    pass


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    list_id: Optional[int] = None
    is_completed: Optional[bool] = None
    is_favorite: Optional[bool] = None
    due_date: Optional[str] = None
    memo: Optional[str] = None
    time: Optional[int] = None


class TaskResponse(BaseModel):
    id: int
    title: str
    list_id: int
    is_completed: bool
    is_favorite: bool
    due_date: Optional[str]
    memo: str
    time: int

    class Config:
        from_attributes = True


class ListBase(BaseModel):
    name: str


class ListCreate(ListBase):
    pass


class ListResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True
