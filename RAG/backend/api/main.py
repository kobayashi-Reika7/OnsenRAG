"""
温泉相談チャットAPI - FastAPIバックエンド（多言語対応）
=======================================================
温泉RAGシステムのWeb API。
React UIからのリクエストを受け付け、RAGで回答を生成して返す。

構成：
  [React UI] → [FastAPI] → [OnsenRAG] → [ChromaDB + LLM]

起動方法（プロジェクトルートから）：
  uvicorn backend.api.main:app --reload --port 8000

エンドポイント：
  POST /api/ask    - 質問を受け付けて回答を返す
  GET  /api/health - ヘルスチェック
"""

import os
import re
import sys
import time
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# backend/ ディレクトリをパスに追加（src パッケージのインポートのため）
_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
# プロジェクトルート（RAG/）- frontend などの参照用
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _BACKEND_DIR)

from src.onsen_rag import OnsenRAG
from src.support_bot import SupportBot
from src import firestore_service
from src.prompts import load_prompt
from src.config import SUPPORTED_LANGS, DEFAULT_LANG

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# リトライ・タイムアウト定数
MAX_RETRIES = 3
RETRY_DELAY_SEC = 1.0
LLM_TIMEOUT_SEC = 60

# 入力サニタイズ: 最大文字数
MAX_QUESTION_LENGTH = 500

# CORS許可オリジン（本番環境では環境変数で制御）
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:8000"
).split(",")
# file:// プロトコルのサポート（ローカル開発用）
if os.getenv("ALLOW_FILE_ORIGIN", "true").lower() == "true":
    ALLOWED_ORIGINS.append("null")  # file:// は Origin: null として送信される


# ============================
# エラーメッセージ辞書（多言語）
# ============================
ERROR_MESSAGES = {
    "ja": {
        "empty_question": "質問が空です。質問を入力してください。",
        "too_long": "質問は{max}文字以内にしてください。（現在{current}文字）",
        "rag_not_ready": "RAGシステムが初期化されていません。しばらくしてから再度お試しください。",
        "timeout": "回答の生成がタイムアウトしました。しばらくしてから再度お試しください。",
        "general_error": "回答の生成に失敗しました。しばらくしてから再度お試しください。",
        "frontend_not_found": "frontend/index.html not found",
    },
    "en": {
        "empty_question": "Question is empty. Please enter a question.",
        "too_long": "Question must be {max} characters or fewer. (Currently {current} characters)",
        "rag_not_ready": "RAG system is not initialized. Please try again later.",
        "timeout": "Answer generation timed out. Please try again later.",
        "general_error": "Failed to generate an answer. Please try again later.",
        "frontend_not_found": "frontend/index.html not found",
    },
}


def _get_error(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    """言語別エラーメッセージを取得する。"""
    msgs = ERROR_MESSAGES.get(lang, ERROR_MESSAGES[DEFAULT_LANG])
    msg = msgs.get(key, ERROR_MESSAGES[DEFAULT_LANG].get(key, key))
    return msg.format(**kwargs) if kwargs else msg


# ============================
# Lifespan（起動・終了の管理）
# ============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan コンテキストマネージャ

    起動時に各言語のRAGシステムを初期化し、終了時にリソースを解放する。
    """
    # --- Startup ---
    print("[START] RAG system initializing (bilingual)...")
    app.state.rag_systems = {}
    app.state.support_bots = {}

    for lang in SUPPORTED_LANGS:
        try:
            rag = OnsenRAG(chunk_size=600, chunk_overlap=75, lang=lang)
            rag.load_from_data_folder()
            bot = SupportBot(
                rag_query_fn=lambda q, _rag=rag: _rag.query(q, k=3),
                enable_escalation=True,
                lang=lang,
            )
            app.state.rag_systems[lang] = rag
            app.state.support_bots[lang] = bot
            logger.info("RAG system initialized successfully (lang=%s)", lang)
        except Exception as error:
            logger.error("RAG initialization failed (lang=%s): %s", lang, error)
            app.state.rag_systems[lang] = None
            app.state.support_bots[lang] = None

    # 後方互換: 既存コードが app.state.rag_system を参照する場合
    app.state.rag_system = app.state.rag_systems.get(DEFAULT_LANG)
    app.state.support_bot = app.state.support_bots.get(DEFAULT_LANG)

    if not any(r for r in app.state.rag_systems.values() if r is not None):
        logger.warning("API will start but no RAG systems are available")

    yield  # アプリケーション実行中

    # --- Shutdown ---
    logger.info("RAG system shutting down...")
    app.state.rag_systems = {}
    app.state.support_bots = {}
    app.state.rag_system = None
    app.state.support_bot = None


# FastAPIアプリケーションの作成
app = FastAPI(
    title="OnsenRAG API",
    description="Hot spring information RAG system Web API (bilingual ja/en)",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS設定（環境変数で制御可能）
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============================
# リクエスト・レスポンスモデル
# ============================
class ChatMessage(BaseModel):
    """会話履歴の1メッセージ"""
    role: str
    content: str


class QuestionRequest(BaseModel):
    """
    質問リクエストのデータ構造

    Attributes:
        question: ユーザーが入力した質問文（最大500文字）
        history: 直近の会話履歴（オプション）
        session_id: セッションID（Firestore保存用、オプション）
        lang: 言語コード ("ja" | "en")
    """
    question: str
    history: list[ChatMessage] = []
    session_id: str = ""
    lang: str = DEFAULT_LANG

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        """入力サニタイズ: 制御文字除去・長さ制限"""
        v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v)
        v = v.strip()
        if not v:
            raise ValueError("Question is empty.")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"Question must be {MAX_QUESTION_LENGTH} characters or fewer. "
                f"(Currently {len(v)} characters)"
            )
        return v

    @field_validator("lang")
    @classmethod
    def validate_lang(cls, v: str) -> str:
        """言語コードのバリデーション"""
        v = v.strip().lower()
        if v not in SUPPORTED_LANGS:
            return DEFAULT_LANG
        return v


class AnswerResponse(BaseModel):
    """
    回答レスポンスのデータ構造
    """
    answer: str
    sources: list[str] = []
    needs_escalation: bool = False
    response_time_ms: int = 0


# ============================
# ヘルパー関数
# ============================
# 温泉地名キーワード（会話履歴から温泉地を検出するため）（言語別）
_LOCATION_NAMES = {
    "ja": {
        "草津": "草津温泉",
        "有馬": "有馬温泉",
        "別府": "別府温泉",
        "箱根": "箱根温泉",
    },
    "en": {
        "Kusatsu": "Kusatsu Onsen",
        "Arima": "Arima Onsen",
        "Beppu": "Beppu Onsen",
        "Hakone": "Hakone Onsen",
    },
}


def _resolve_question_simple(
    question: str,
    history: list[ChatMessage],
    lang: str = DEFAULT_LANG,
) -> str:
    """ルールベースのクエリ補完（フォールバック用）。"""
    loc_names = _LOCATION_NAMES.get(lang, _LOCATION_NAMES[DEFAULT_LANG])

    if any(name in question for name in loc_names):
        return question

    if len(question) > 15:
        return question

    if not history:
        return question

    connector = "の" if lang == "ja" else " "
    for msg in reversed(history):
        for keyword, full_name in loc_names.items():
            if keyword in msg.content:
                return f"{full_name}{connector}{question}"

    return question


def _resolve_question(
    question: str,
    history: list[ChatMessage],
    lang: str = DEFAULT_LANG,
) -> str:
    """
    会話履歴から文脈を読み取り、曖昧なクエリを自己完結した質問に書き換える。

    LLMベースの解決を試み、失敗時はルールベースにフォールバック。
    """
    if not history:
        return question

    loc_names = _LOCATION_NAMES.get(lang, _LOCATION_NAMES[DEFAULT_LANG])
    if any(name in question for name in loc_names):
        return question

    if len(question) > 20:
        return question

    # LLMベースの質問解決を試行
    rag = _get_rag_for_lang(lang, raise_error=False)
    query_resolve_prompt = load_prompt("query_resolve.txt", lang)
    if rag is None or not query_resolve_prompt:
        return _resolve_question_simple(question, history, lang)

    try:
        from langchain_core.prompts import PromptTemplate
        from src.llm_factory import invoke_with_retry

        recent_history = history[-5:]
        history_text = "\n".join(
            f"{msg.role}: {msg.content}" for msg in recent_history
        )

        prompt = PromptTemplate(
            template=query_resolve_prompt,
            input_variables=["history", "question"],
        )
        chain = prompt | rag.llm
        resolved = invoke_with_retry(chain, {
            "history": history_text,
            "question": question,
        }, max_retries=1)

        resolved = resolved.strip()
        if resolved and len(resolved) < MAX_QUESTION_LENGTH and resolved != question:
            logger.info("LLM query resolved: '%s' -> '%s'", question, resolved)
            return resolved

    except Exception as e:
        logger.warning("LLM query resolution failed (falling back to rule-based): %s", str(e)[:80])

    return _resolve_question_simple(question, history, lang)


def _get_rag_for_lang(lang: str = DEFAULT_LANG, raise_error: bool = True) -> OnsenRAG | None:
    """言語別 RAG システムを取得。"""
    rag_systems = getattr(app.state, "rag_systems", {})
    rag = rag_systems.get(lang)
    if (rag is None or rag.vectorstore is None) and raise_error:
        raise HTTPException(
            status_code=503,
            detail=_get_error("rag_not_ready", lang),
        )
    return rag


def _get_bot_for_lang(lang: str = DEFAULT_LANG) -> SupportBot | None:
    """言語別サポートボットを取得。"""
    bots = getattr(app.state, "support_bots", {})
    return bots.get(lang)


# 後方互換ヘルパー
def _get_rag() -> OnsenRAG:
    return _get_rag_for_lang(DEFAULT_LANG)


def _get_bot() -> SupportBot | None:
    return _get_bot_for_lang(DEFAULT_LANG)


# ============================
# エンドポイント
# ============================
@app.get("/api/health")
async def health_check():
    """ヘルスチェック"""
    rag_systems = getattr(app.state, "rag_systems", {})
    lang_status = {}
    for lang in SUPPORTED_LANGS:
        rag = rag_systems.get(lang)
        lang_status[lang] = rag is not None and rag.vectorstore is not None

    is_ready = any(lang_status.values())

    return {
        "status": "ok" if is_ready else "not_ready",
        "languages": lang_status,
        "firestore_available": firestore_service.is_available(),
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    質問を受け付けてRAGで回答を生成（多言語対応）
    """
    lang = request.lang
    rag = _get_rag_for_lang(lang)
    bot = _get_bot_for_lang(lang)
    start_time = time.time()
    sid = request.session_id

    # 会話履歴からクエリを補完
    resolved_question = _resolve_question(request.question, request.history, lang)
    if resolved_question != request.question:
        logger.info("Query resolved: '%s' -> '%s'", request.question, resolved_question)

    # ユーザーメッセージを Firestore に保存
    if sid:
        firestore_service.save_message(sid, "user", request.question)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if bot is not None:
                def _sync_support_ask():
                    return bot.ask(resolved_question, k=3)

                loop = asyncio.get_event_loop()
                resp = await asyncio.wait_for(
                    loop.run_in_executor(None, _sync_support_ask),
                    timeout=LLM_TIMEOUT_SEC,
                )
                elapsed_ms = int((time.time() - start_time) * 1000)
                answer_resp = AnswerResponse(
                    answer=resp.answer,
                    sources=resp.sources,
                    needs_escalation=resp.needs_escalation,
                    response_time_ms=elapsed_ms,
                )
                _save_to_firestore(sid, request.question, resolved_question, answer_resp)
                return answer_resp

            # SupportBot未初期化時のフォールバック
            def _sync_query():
                return rag.query(resolved_question, k=3)

            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_query),
                timeout=LLM_TIMEOUT_SEC,
            )
            sources = []
            if "source_documents" in result:
                sources = [doc.page_content[:200] for doc in result["source_documents"]]
            answer = result["result"]
            if hasattr(answer, "content"):
                answer = str(answer.content)

            elapsed_ms = int((time.time() - start_time) * 1000)
            answer_resp = AnswerResponse(
                answer=answer.strip(),
                sources=sources,
                response_time_ms=elapsed_ms,
            )
            _save_to_firestore(sid, request.question, resolved_question, answer_resp)
            return answer_resp

        except asyncio.TimeoutError as error:
            last_error = error
            logger.warning("Answer generation timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_SEC)
        except Exception as error:
            last_error = error
            logger.warning("Answer generation error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, error)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_SEC)

    if isinstance(last_error, asyncio.TimeoutError):
        detail = _get_error("timeout", lang)
    else:
        detail = _get_error("general_error", lang)
        logger.error("Final answer generation error: %s", last_error)

    raise HTTPException(status_code=500, detail=detail)


def _save_to_firestore(
    session_id: str,
    question: str,
    resolved_question: str,
    resp: AnswerResponse,
) -> None:
    """回答をFirestoreに保存（失敗してもAPIレスポンスには影響しない）"""
    if not session_id:
        return
    try:
        firestore_service.save_message(session_id, "assistant", resp.answer)
        firestore_service.save_qa_log(
            session_id=session_id,
            question=question,
            resolved_question=resolved_question,
            answer=resp.answer,
            sources=resp.sources,
            response_time_ms=resp.response_time_ms,
            needs_escalation=resp.needs_escalation,
        )
    except Exception as e:
        logger.warning("Firestore save error (continuing): %s", e)


@app.get("/")
async def serve_frontend():
    """フロントエンドHTMLを配信"""
    frontend_path = os.path.join(_PROJECT_ROOT, "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {"message": _get_error("frontend_not_found")}


# --- 静的ファイルのマウント（CSS / JS） ---
_frontend_dir = os.path.join(_PROJECT_ROOT, "frontend")
_css_dir = os.path.join(_frontend_dir, "css")
_js_dir = os.path.join(_frontend_dir, "js")

if os.path.isdir(_css_dir):
    app.mount("/css", StaticFiles(directory=_css_dir), name="css")
if os.path.isdir(_js_dir):
    app.mount("/js", StaticFiles(directory=_js_dir), name="js")


@app.post("/api/search")
async def search_chunks(request: QuestionRequest):
    """検索結果のみを返す（評価・デバッグ用）"""
    lang = request.lang
    rag = _get_rag_for_lang(lang)
    docs = rag.search_chunks(request.question, k=3)

    return {
        "question": request.question,
        "lang": lang,
        "chunks": [
            {
                "content": doc.page_content,
                "length": len(doc.page_content),
            }
            for doc in docs
        ]
    }
