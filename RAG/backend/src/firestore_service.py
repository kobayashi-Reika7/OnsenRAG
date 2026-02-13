"""
Firestore サービス
===================
Firebase Firestore を使った会話履歴・Q&Aログの永続化。

Firestoreコレクション構成:
  conversations/{session_id}/messages/{auto_id}
    - role: "user" | "assistant"
    - content: メッセージ本文
    - created_at: タイムスタンプ

  qa_logs/{auto_id}
    - session_id: セッションID
    - question: ユーザーの元の質問
    - resolved_question: コンテキスト補完後の質問
    - answer: RAGの回答
    - sources: 参照チャンク（配列）
    - response_time_ms: 応答時間
    - needs_escalation: エスカレーション提案有無
    - created_at: タイムスタンプ

環境変数:
  FIREBASE_PROJECT_ID     - GCPプロジェクトID（必須）
  GOOGLE_APPLICATION_CREDENTIALS - サービスアカウントキーのパス（ローカル開発用）
  ※ Cloud Run 上ではデフォルト認証が自動適用されるため不要
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Firebase Admin SDK（遅延インポート: 未インストール時もエラーにしない）
_db = None
_initialized = False


def _get_db():
    """Firestore クライアントを取得（遅延初期化・シングルトン）"""
    global _db, _initialized

    if _initialized:
        return _db

    _initialized = True

    project_id = os.getenv("FIREBASE_PROJECT_ID", "")
    if not project_id:
        logger.warning(
            "FIREBASE_PROJECT_ID が未設定のため Firestore は無効です。"
            "ローカルモードで動作します。"
        )
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        # 既に初期化済みの場合はスキップ
        if not firebase_admin._apps:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {"projectId": project_id})
            else:
                # Cloud Run ではデフォルト認証を使用
                firebase_admin.initialize_app(options={"projectId": project_id})

        _db = firestore.client()
        logger.info("Firestore 初期化完了 (project: %s)", project_id)
        return _db

    except ImportError:
        logger.warning("firebase-admin がインストールされていません。Firestore は無効です。")
        return None
    except Exception as e:
        logger.error("Firestore 初期化エラー: %s", e)
        return None


# ============================
# 会話履歴
# ============================
def save_message(session_id: str, role: str, content: str) -> Optional[str]:
    """
    会話メッセージを Firestore に保存

    Args:
        session_id: セッションID（フロントエンドで生成）
        role: "user" | "assistant"
        content: メッセージ本文

    Returns:
        ドキュメントID（保存成功時）、None（Firestore無効時）
    """
    db = _get_db()
    if db is None:
        return None

    try:
        doc_ref = (
            db.collection("conversations")
            .document(session_id)
            .collection("messages")
            .document()
        )
        doc_ref.set({
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc),
        })
        return doc_ref.id
    except Exception as e:
        logger.error("メッセージ保存エラー: %s", e)
        return None


def get_messages(session_id: str, limit: int = 20) -> list[dict]:
    """
    セッションの会話履歴を取得

    Args:
        session_id: セッションID
        limit: 取得件数上限

    Returns:
        メッセージのリスト [{"role": ..., "content": ..., "created_at": ...}, ...]
    """
    db = _get_db()
    if db is None:
        return []

    try:
        messages_ref = (
            db.collection("conversations")
            .document(session_id)
            .collection("messages")
            .order_by("created_at")
            .limit(limit)
        )
        return [doc.to_dict() for doc in messages_ref.stream()]
    except Exception as e:
        logger.error("メッセージ取得エラー: %s", e)
        return []


# ============================
# Q&A ログ
# ============================
def save_qa_log(
    session_id: str,
    question: str,
    resolved_question: str,
    answer: str,
    sources: list[str],
    response_time_ms: int,
    needs_escalation: bool = False,
) -> Optional[str]:
    """
    質問・回答のログを Firestore に保存

    Returns:
        ドキュメントID（保存成功時）、None（Firestore無効時）
    """
    db = _get_db()
    if db is None:
        return None

    try:
        doc_ref = db.collection("qa_logs").document()
        doc_ref.set({
            "session_id": session_id,
            "question": question,
            "resolved_question": resolved_question,
            "answer": answer,
            "sources": sources,
            "response_time_ms": response_time_ms,
            "needs_escalation": needs_escalation,
            "created_at": datetime.now(timezone.utc),
        })
        return doc_ref.id
    except Exception as e:
        logger.error("Q&Aログ保存エラー: %s", e)
        return None


def get_qa_logs(limit: int = 50) -> list[dict]:
    """直近のQ&Aログを取得"""
    db = _get_db()
    if db is None:
        return []

    try:
        logs_ref = (
            db.collection("qa_logs")
            .order_by("created_at", direction="DESCENDING")
            .limit(limit)
        )
        return [doc.to_dict() for doc in logs_ref.stream()]
    except Exception as e:
        logger.error("Q&Aログ取得エラー: %s", e)
        return []


def is_available() -> bool:
    """Firestore が利用可能かどうか"""
    return _get_db() is not None
