"""
FastAPI バックエンド - 予約アプリ用 API
- ユーザー同期・認証
- 空き枠取得・予約作成（勤務判定・医師割当はすべてバックエンド）
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# backend/.env を読み込み（GOOGLE_CLOUD_PROJECT 等の環境変数を設定）
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, HTTPException, Header

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from models import UserResponse, SyncUserBody, SlotItem, AvailabilityForDateResponse, CreateReservationBody, ReservationCreated
import store
from firebase_admin_client import verify_id_token
from reservation_service import get_availability_for_date, create_reservation as create_reservation_service

# CORS: フロントエンド（Vite 開発サーバー）を許可
# 5200 が使用中だと Vite が 5201 を使うため、5201 も許可する
_default_origins = ["http://localhost:5200", "http://127.0.0.1:5200", "http://localhost:5201", "http://127.0.0.1:5201"]
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
_extra_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] if _origins_env else []
ALLOWED_ORIGINS = list(dict.fromkeys([*_default_origins, *_extra_origins]))

app = FastAPI(title="Reservation API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """ヘルスチェック"""
    return {"status": "ok"}


@app.get("/debug/cors")
def debug_cors():
    """デバッグ用: CORS 許可オリジン一覧を返す（ローカル確認用）"""
    return {"allowed_origins": ALLOWED_ORIGINS}


def _get_bearer_token(authorization: str | None) -> str:
    """Authorization ヘッダーから Bearer トークンを抽出。401 時は原因をログ出力。"""
    if not authorization:
        logger.warning("[401] Authorization ヘッダーが未指定です。")
        raise HTTPException(status_code=401, detail="Authorization ヘッダーが必要です。")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        logger.warning("[401] Authorization の形式が不正です（Bearer プレフィックスなし）。")
        raise HTTPException(status_code=401, detail="Authorization: Bearer <token> の形式で送信してください。")
    token = authorization[len(prefix):].strip()
    if not token:
        logger.warning("[401] IDトークンが空です。")
        raise HTTPException(status_code=401, detail="IDトークンが空です。")
    return token


@app.get("/users/me", response_model=UserResponse)
def users_me(authorization: str | None = Header(default=None)):
    """
    Firebase IDトークンを検証して、ユーザー情報（uid/email）を返す。
    ついでにバックエンド側メモリDBへ upsert（確認用）。
    """
    token = _get_bearer_token(authorization)
    try:
        claims = verify_id_token(token)
    except Exception as e:
        logger.warning("[401] GET /users/me IDトークン検証失敗: %s", e)
        raise HTTPException(status_code=401, detail="IDトークンの検証に失敗しました。") from e

    uid = str(claims.get("uid", ""))
    email = str(claims.get("email", ""))
    if not uid:
        logger.warning("[401] GET /users/me トークンから uid を取得できません。")
        raise HTTPException(status_code=401, detail="トークンから uid を取得できません。")
    store.upsert_user(uid, email)
    return {"uid": uid, "email": email}


@app.post("/users/sync", response_model=dict)
def sync_user(body: SyncUserBody):
    """
    フロントから明示同期する（任意）。
    ※認証を厳密にするなら /users/me のみを使い、sync は廃止する。
    """
    store.upsert_user(body.uid, body.email)
    return {"ok": True}


@app.get("/users", response_model=list[UserResponse])
def list_users():
    """ユーザー一覧（管理・確認用）"""
    return store.get_all_users()


# ----- 予約・空き枠 API（業務ロジックはバックエンド専用） -----


@app.get("/api")
def api_info():
    """Day5 予約 API であることを示す（404 時に別サーバーが動いていないか確認用）"""
    return {
        "name": "Day5 Reservation API",
        "endpoints": {
            "slots": "GET /api/slots",
            "reservations": "POST /api/reservations",
        },
    }


@app.get("/api/slots", response_model=AvailabilityForDateResponse)
def api_slots(department: str = "", date: str = "", authorization: str | None = Header(default=None)):
    """
    診療科・日付の空き枠を返す。祝日・過去日はバックエンドで判定し date, is_holiday, reason を含める。
    フロントは祝日判定を行わず、このレスポンスのみで表示する。
    """
    department = (department or "").strip()
    date = (date or "").strip()
    try:
        return get_availability_for_date(department, date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/reservations", response_model=ReservationCreated)
def api_create_reservation(body: CreateReservationBody, authorization: str | None = Header(default=None)):
    """
    予約を確定する。担当医はバックエンドで自動割当。
    認証必須。
    """
    token = _get_bearer_token(authorization)
    try:
        claims = verify_id_token(token)
    except Exception as e:
        logger.warning("[401] IDトークン検証失敗: %s", e)
        raise HTTPException(status_code=401, detail="IDトークンの検証に失敗しました。") from e
    uid = str(claims.get("uid", ""))
    if not uid:
        logger.warning("[401] トークンから uid を取得できません。")
        raise HTTPException(status_code=401, detail="トークンから uid を取得できません。")

    department = (body.department or "").strip()
    date = (body.date or "").strip()
    time = (body.time or "").strip()
    if not department or not date or not time:
        raise HTTPException(status_code=400, detail="診療科・日付・時間は必須です。")
    try:
        out = create_reservation_service(department, date, time, uid)
        return ReservationCreated(id=out["id"], date=out["date"], time=out["time"], department=out["departmentId"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("POST /api/reservations failed: %s", e)
        raise HTTPException(status_code=500, detail="予約の保存に失敗しました。") from e
