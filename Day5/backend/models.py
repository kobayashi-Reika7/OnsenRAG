"""
Pydantic モデル（ユーザー・予約）
認証は Firebase に一本化するため、バックエンドは ID トークン検証後のユーザー情報のみ扱う
"""
from pydantic import BaseModel, Field
from typing import Optional


class UserResponse(BaseModel):
    """ユーザー情報レスポンス（Firebase uid ベース）"""
    uid: str
    email: str

    class Config:
        from_attributes = True


class SlotItem(BaseModel):
    """空き枠1件（フロントは表示のみ、判定はバックエンド）"""
    time: str
    reservable: bool


class AvailabilityForDateResponse(BaseModel):
    """1日分の空き状況（祝日・過去日・理由をバックエンドで判定、フロントは表示のみ）"""
    date: str
    is_holiday: bool
    reservable: bool
    reason: Optional[str] = None  # "holiday" | "past" | "closed" | null
    slots: list[SlotItem]


class CreateReservationBody(BaseModel):
    """予約作成（診療科・日・時間・種別。担当医はバックエンドで自動割当）"""
    department: str = Field(..., min_length=1, max_length=100)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    time: str = Field(..., pattern=r"^\d{2}:\d{2}$")
    purpose: str = Field(default="", max_length=20)  # 初診/再診


class ReservationCreated(BaseModel):
    """予約作成レスポンス（UIには doctorId を出さない）"""
    id: str
    date: str
    time: str
    department: str
