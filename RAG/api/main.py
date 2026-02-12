"""
温泉相談チャットAPI - FastAPIバックエンド
==========================================
温泉RAGシステムのWeb API。
React UIからのリクエストを受け付け、RAGで回答を生成して返す。

構成：
  [React UI] → [FastAPI] → [OnsenRAG] → [ChromaDB + LLM]

起動方法：
  uvicorn api.main:app --reload --port 8000

エンドポイント：
  POST /api/ask    - 質問を受け付けて回答を返す
  GET  /api/health - ヘルスチェック
"""

import os
import sys
import time
import logging
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# プロジェクトルートをパスに追加（srcのインポートのため）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.onsen_rag import OnsenRAG
from src.support_bot import SupportBot

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# リトライ・タイムアウト定数
MAX_RETRIES = 3
RETRY_DELAY_SEC = 1.0
LLM_TIMEOUT_SEC = 60

# FastAPIアプリケーションの作成
app = FastAPI(
    title="OnsenRAG API",
    description="温泉情報RAGシステムのWeb API",
    version="1.0.0"
)

# CORS設定（React開発サーバーからのアクセスを許可）
# なぜ必要か：React（localhost:3000）からFastAPI（localhost:8000）への
# クロスオリジンリクエストはブラウザがブロックするため
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ローカル開発用（file://含む全オリジン許可）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAGシステムのグローバルインスタンス
rag_system: OnsenRAG = None
support_bot: SupportBot = None  # カスタマーサポートボット（問い合わせ特化）


class QuestionRequest(BaseModel):
    """
    質問リクエストのデータ構造

    Attributes:
        question: ユーザーが入力した質問文
    """
    question: str


class AnswerResponse(BaseModel):
    """
    回答レスポンスのデータ構造

    Attributes:
        answer: RAGが生成した回答テキスト（回答+根拠チャンクID形式）
        sources: 参照したチャンクの内容（根拠の可視化用）
        chunk_ids: 参照したチャンクIDリスト（根拠のトレーサビリティ用）
        needs_escalation: 担当者へのおつなぎを提案するか（サポートボット時）
    """
    answer: str
    sources: list[str] = []
    chunk_ids: list[str] = []
    needs_escalation: bool = False


@app.on_event("startup")
async def startup_event():
    """
    アプリ起動時にRAGシステムを初期化

    温泉テキストデータを読み込み、ベクトルDBを構築する。
    この処理は起動時に一度だけ実行される。
    """
    global rag_system, support_bot

    print("[START] RAG system initializing...")
    support_bot = None

    try:
        rag_system = OnsenRAG(chunk_size=600, chunk_overlap=75)
        rag_system.load_from_data_folder()
        # サポートボット（エスカレーション提案付き）
        support_bot = SupportBot(
            rag_query_fn=lambda q: rag_system.query(q, k=3),
            enable_escalation=True,
        )
        logger.info("RAG system initialized successfully")
    except Exception as error:
        logger.error("RAG initialization failed: %s", error)
        logger.warning("API will start but questions cannot be answered")


@app.get("/api/health")
async def health_check():
    """
    ヘルスチェックエンドポイント

    RAGシステムの状態を確認する。
    UIから接続テストに使用。
    """
    is_ready = rag_system is not None \
        and rag_system.vectorstore is not None

    return {
        "status": "ok" if is_ready else "not_ready",
        "rag_initialized": is_ready,
        "sources": [
            "kusatsu_chunks.json",
            "hakone_chunks.json",
            "beppu_chunks.json",
            "arima_chunks.json",
            "onsen_knowledge_chunks.json",
        ] if is_ready else [],
    }


async def _run_query_sync(question: str, k: int):
    """同期RAGを非同期で実行（タイムアウト対応）"""
    def _query():
        return rag_system.query(question, k=k)

    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _query),
        timeout=LLM_TIMEOUT_SEC,
    )


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    質問を受け付けてRAGで回答を生成

    処理の流れ：
    1. リクエストから質問文を取得
    2. OnsenRAGで関連チャンクを検索
    3. LLMで回答を生成
    4. 回答と参照ソースをレスポンスとして返す

    Args:
        request: 質問リクエスト（questionフィールド）

    Returns:
        AnswerResponse: 回答と参照ソース
    """
    # RAGが初期化されていない場合はエラー
    if rag_system is None or rag_system.vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="RAGシステムが初期化されていません。"
                   "APIキーの設定を確認してください。"
        )

    # 空の質問はエラー
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="質問が空です。質問を入力してください。"
        )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            # サポートボット経由で回答（エスカレーション提案付き）
            if support_bot is not None:
                def _sync_support_ask():
                    return support_bot.ask(request.question, k=3)

                loop = asyncio.get_event_loop()
                resp = await asyncio.wait_for(
                    loop.run_in_executor(None, _sync_support_ask),
                    timeout=LLM_TIMEOUT_SEC,
                )
                return AnswerResponse(
                    answer=resp.answer,
                    sources=resp.sources,
                    chunk_ids=resp.chunk_ids,
                    needs_escalation=resp.needs_escalation,
                )

            # 従来のRAG直接呼び出し（サポートボット未初期化時）
            result = await _run_query_sync(request.question, k=3)
            sources = []
            chunk_ids = result.get("chunk_ids", [])
            if "source_documents" in result:
                sources = [doc.page_content[:200] for doc in result["source_documents"]]
            answer = result["result"]
            if hasattr(answer, "content"):
                answer = str(answer.content)
            return AnswerResponse(
                answer=answer.strip(),
                sources=sources,
                chunk_ids=chunk_ids,
            )

        except asyncio.TimeoutError as error:
            last_error = error
            logger.warning("回答生成タイムアウト (試行 %d/%d)", attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_SEC)
        except Exception as error:
            last_error = error
            logger.warning("回答生成エラー (試行 %d/%d): %s", attempt + 1, MAX_RETRIES, error)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_SEC)

    msg = "タイムアウト" if isinstance(last_error, asyncio.TimeoutError) else str(last_error)
    raise HTTPException(
        status_code=500,
        detail=f"回答の生成に失敗しました（{MAX_RETRIES}回試行）: {msg}"
    )


@app.get("/")
async def serve_frontend():
    """
    フロントエンドHTMLを配信

    http://localhost:8000/ でチャットUIにアクセスできる。
    """
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "frontend", "index.html"
    )
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {"message": "frontend/index.html が見つかりません"}


@app.post("/api/search")
async def search_chunks(request: QuestionRequest):
    """
    検索結果のみを返す（評価・デバッグ用）

    LLMの回答生成は行わず、検索されたチャンクのみを返す。
    RAGの検索精度を確認したい時に使用。
    """
    if rag_system is None or rag_system.vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="RAGシステムが初期化されていません。"
        )

    docs = rag_system.search_chunks(request.question, k=3)

    return {
        "question": request.question,
        "chunks": [
            {
                "content": doc.page_content,
                "length": len(doc.page_content)
            }
            for doc in docs
        ]
    }
