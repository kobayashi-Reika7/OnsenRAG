"""
FastAPI バックエンド - 予約アプリ用 API
- ユーザー認証（Firebase IDトークン検証）
- 空き枠取得・予約作成（勤務判定・医師割当はすべてバックエンド）
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# backend/.env を読み込み（GOOGLE_CLOUD_PROJECT 等の環境変数を設定）
load_dotenv(Path(__file__).resolve().parent / ".env")

# ログ設定（タイムスタンプ・レベル付き）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from fastapi import FastAPI, HTTPException, Header, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from models import UserResponse, SlotItem, AvailabilityForDateResponse, CreateReservationBody, ReservationCreated
from firebase_admin_client import verify_id_token
from reservation_service import get_availability_for_date, get_availability_for_dates, create_reservation as create_reservation_service, cancel_reservation as cancel_reservation_service

# CORS: フロントエンド（Vite 開発サーバー）を許可
_default_origins = ["http://localhost:5200", "http://127.0.0.1:5200", "http://localhost:5201", "http://127.0.0.1:5201"]
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
_extra_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] if _origins_env else []
ALLOWED_ORIGINS = list(dict.fromkeys([*_default_origins, *_extra_origins]))


# --------------- レート制限ミドルウェア ---------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """IPアドレス単位の簡易レート制限（単一サーバー向け）"""
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        self._requests[client_ip] = [t for t in self._requests[client_ip] if now - t < self.window]
        if len(self._requests[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "リクエストが多すぎます。しばらくしてから再度お試しください。"},
            )
        self._requests[client_ip].append(now)
        return await call_next(request)


app = FastAPI(title="Reservation API", version="1.0")

app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
def health():
    """ヘルスチェック"""
    return {"status": "ok"}


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
    """Firebase IDトークンを検証して、ユーザー情報（uid/email）を返す。"""
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
    return {"uid": uid, "email": email}


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


def _try_get_uid(authorization: str | None) -> str:
    """Authorization ヘッダーからユーザーIDを取得（オプション）。失敗時は空文字を返す。"""
    if not authorization:
        return ""
    try:
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            return ""
        token = authorization[len(prefix):].strip()
        if not token:
            return ""
        claims = verify_id_token(token)
        return str(claims.get("uid", ""))
    except Exception:
        return ""


@app.get("/api/slots/week")
def api_slots_week(department: str = "", dates: str = "", authorization: str | None = Header(default=None)):
    """
    複数日分の空き枠を一括で返す（高速版）。
    dates はカンマ区切り（例: 2026-02-10,2026-02-11,...）。最大14日。
    認証トークンがある場合、そのユーザーの予約済みスロットも×にする。
    """
    department = (department or "").strip()
    date_list = [d.strip() for d in (dates or "").split(",") if d.strip()]
    if not department or not date_list:
        return []
    if len(date_list) > 14:
        date_list = date_list[:14]
    uid = _try_get_uid(authorization)
    try:
        return get_availability_for_dates(department, date_list, user_id=uid)
    except Exception as e:
        logger.exception("GET /api/slots/week failed: %s", e)
        raise HTTPException(status_code=500, detail="空き枠情報の取得に失敗しました。") from e


@app.get("/api/slots", response_model=AvailabilityForDateResponse)
def api_slots(department: str = "", date: str = "", authorization: str | None = Header(default=None)):
    """
    診療科・日付の空き枠を返す。祝日・過去日はバックエンドで判定し date, is_holiday, reason を含める。
    認証トークンがある場合、そのユーザーの予約済みスロットも×にする。
    """
    department = (department or "").strip()
    date = (date or "").strip()
    uid = _try_get_uid(authorization)
    try:
        return get_availability_for_date(department, date, user_id=uid)
    except Exception as e:
        logger.exception("GET /api/slots failed: %s", e)
        raise HTTPException(status_code=500, detail="空き枠情報の取得に失敗しました。") from e


@app.post("/api/reservations", response_model=ReservationCreated)
def api_create_reservation(body: CreateReservationBody, authorization: str | None = Header(default=None)):
    """予約を確定する。担当医はバックエンドで自動割当。認証必須。"""
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
    purpose = (body.purpose or "").strip()
    if not department or not date or not time:
        raise HTTPException(status_code=400, detail="診療科・日付・時間は必須です。")
    try:
        out = create_reservation_service(department, date, time, uid, purpose=purpose)
        return ReservationCreated(id=out["id"], date=out["date"], time=out["time"], department=out["departmentId"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("POST /api/reservations failed: %s", e)
        raise HTTPException(status_code=500, detail="予約の保存に失敗しました。") from e


@app.delete("/api/reservations/{reservation_id}")
def api_cancel_reservation(reservation_id: str, authorization: str | None = Header(default=None)):
    """
    予約をキャンセルする。認証必須。
    booked_slots のスロットも同時に解放し、ダブルブッキングを防止する。
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

    if not reservation_id or not reservation_id.strip():
        raise HTTPException(status_code=400, detail="予約IDが必要です。")

    try:
        result = cancel_reservation_service(uid, reservation_id.strip())
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("DELETE /api/reservations/%s failed: %s", reservation_id, e)
        raise HTTPException(status_code=500, detail="予約のキャンセルに失敗しました。") from e
