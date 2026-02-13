"""
SupportBot - カスタマーサポートボット（多言語対応）
=====================================================
RAGをベースに、問い合わせ特化の機能を追加。

機能:
- FAQパターンによる即答（オプション）
- 回答に自信がない場合のエスカレーション提案
- 担当者おつなぎ案内
"""

from typing import Callable, Optional
from dataclasses import dataclass

from src.config import DEFAULT_LANG


@dataclass
class SupportResponse:
    """サポートボットの応答"""
    answer: str
    needs_escalation: bool  # 担当者へのおつなぎを提案するか
    sources: list[str] = None
    chunk_ids: list[str] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.chunk_ids is None:
            self.chunk_ids = []


class SupportBot:
    """
    カスタマーサポートボット（多言語対応）

    RAGの query 関数をラップし、エスカレーション提案などを付加する。
    """

    # エスカレーションを提案すべきキーワード（言語別）
    ESCALATION_KEYWORDS = {
        "ja": [
            "分かりません",
            "該当情報が見つかりません",
            "参考情報からは",
            "文書内に",
            "判断できません",
            "お答えできません",
            "情報がございません",
        ],
        "en": [
            "I don't know",
            "not available",
            "no information",
            "cannot determine",
            "unable to find",
            "not found in",
            "cannot answer",
        ],
    }

    # FAQパターン（質問 → 即答）（言語別）
    DEFAULT_FAQ = {
        "ja": {
            "お問い合わせ": "お問い合わせありがとうございます。内容を確認の上、担当者よりご連絡いたします。",
            "担当者": "担当者へのおつなぎをご希望の場合、お名前とご用件をお伝えください。",
            "急ぎ": "お急ぎの件は担当者より優先的に対応いたします。",
        },
        "en": {
            "contact": "Thank you for your inquiry. A representative will get back to you shortly.",
            "representative": "If you'd like to speak with a representative, please provide your name and inquiry details.",
            "urgent": "Urgent matters will be prioritized by our representative.",
        },
    }

    # エスカレーション提案テキスト（言語別）
    ESCALATION_SUFFIX = {
        "ja": "\n\n※上記で解決しない場合は、担当者へおつなぎいたします。",
        "en": "\n\nIf this does not resolve your question, we can connect you with a representative.",
    }

    # エラーメッセージ（言語別）
    ERROR_MESSAGE = {
        "ja": "申し訳ございません。エラーが発生しました: {error}\n担当者へおつなぎしますか？",
        "en": "We apologize for the inconvenience. An error occurred: {error}\nWould you like to speak with a representative?",
    }

    def __init__(
        self,
        rag_query_fn: Callable[[str], dict],
        faq: Optional[dict] = None,
        enable_escalation: bool = True,
        lang: str = DEFAULT_LANG,
    ):
        """
        Args:
            rag_query_fn: RAGの query(question) を呼び出す関数
            faq: FAQ辞書 {キーワード: 即答}。None の場合はデフォルトFAQ
            enable_escalation: エスカレーション提案を有効にするか
            lang: 言語コード ("ja" | "en")
        """
        self.rag_query_fn = rag_query_fn
        self.lang = lang
        self.faq = faq or self.DEFAULT_FAQ.get(lang, self.DEFAULT_FAQ["ja"])
        self.enable_escalation = enable_escalation

    def ask(self, question: str, k: int = 3) -> SupportResponse:
        """
        質問に対して回答を生成し、エスカレーション要否を判定
        """
        # FAQパターンマッチ（質問にキーワードが含まれる場合）
        for keyword, faq_answer in self.faq.items():
            if keyword in question:
                return SupportResponse(
                    answer=faq_answer,
                    needs_escalation=False,
                    sources=[],
                    chunk_ids=[],
                )

        # RAGで回答生成
        try:
            result = self.rag_query_fn(question)
        except Exception as e:
            error_msg = self.ERROR_MESSAGE.get(self.lang, self.ERROR_MESSAGE["ja"])
            return SupportResponse(
                answer=error_msg.format(error=str(e)),
                needs_escalation=True,
                sources=[],
                chunk_ids=[],
            )

        answer = result.get("result", "")
        if hasattr(answer, "content"):
            answer = str(answer.content) if answer else ""

        sources = []
        chunk_ids = result.get("chunk_ids", [])
        if "source_documents" in result:
            sources = [
                doc.page_content[:200]
                for doc in result["source_documents"]
            ]

        # エスカレーション要否の判定
        needs_escalation = False
        if self.enable_escalation:
            escalation_kws = self.ESCALATION_KEYWORDS.get(
                self.lang, self.ESCALATION_KEYWORDS["ja"]
            )
            for kw in escalation_kws:
                if kw.lower() in answer.lower():
                    needs_escalation = True
                    suffix = self.ESCALATION_SUFFIX.get(
                        self.lang, self.ESCALATION_SUFFIX["ja"]
                    )
                    answer += suffix
                    break

        return SupportResponse(
            answer=answer.strip(),
            needs_escalation=needs_escalation,
            sources=sources,
            chunk_ids=chunk_ids,
        )
