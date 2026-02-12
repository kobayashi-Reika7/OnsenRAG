"""
SupportBot - カスタマーサポートボット
========================================
RAGをベースに、問い合わせ特化の機能を追加。

機能:
- FAQパターンによる即答（オプション）
- 回答に自信がない場合のエスカレーション提案
- 担当者おつなぎ案内
"""

from typing import Callable, Optional
from dataclasses import dataclass


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
    カスタマーサポートボット

    RAGの query 関数をラップし、エスカレーション提案などを付加する。
    """

    # エスカレーションを提案すべきキーワード（回答に含まれる場合）
    ESCALATION_KEYWORDS = [
        "分かりません",
        "該当情報が見つかりません",
        "参考情報からは",
        "文書内に",
        "判断できません",
    ]

    # FAQパターン（質問 → 即答） オプションで上書き可能
    DEFAULT_FAQ = {
        "お問い合わせ": "お問い合わせありがとうございます。内容を確認の上、担当者よりご連絡いたします。",
        "担当者": "担当者へのおつなぎをご希望の場合、お名前とご用件をお伝えください。",
        "急ぎ": "お急ぎの件は担当者より優先的に対応いたします。",
    }

    def __init__(
        self,
        rag_query_fn: Callable[[str], dict],
        faq: Optional[dict] = None,
        enable_escalation: bool = True,
    ):
        """
        Args:
            rag_query_fn: RAGの query(question) を呼び出す関数。{"result": str, "source_documents": [...]} を返すこと
            faq: FAQ辞書 {キーワード: 即答}。None の場合は DEFAULT_FAQ
            enable_escalation: エスカレーション提案を有効にするか
        """
        self.rag_query_fn = rag_query_fn
        self.faq = faq or self.DEFAULT_FAQ
        self.enable_escalation = enable_escalation

    def ask(self, question: str, k: int = 3) -> SupportResponse:
        """
        質問に対して回答を生成し、エスカレーション要否を判定

        Args:
            question: ユーザーの質問
            k: RAG検索件数

        Returns:
            SupportResponse: 回答とエスカレーション提案フラグ
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
            return SupportResponse(
                answer=f"申し訳ございません。エラーが発生しました: {str(e)}\n"
                       "担当者へおつなぎしますか？",
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
            for kw in self.ESCALATION_KEYWORDS:
                if kw in answer:
                    needs_escalation = True
                    answer += "\n\n※上記で解決しない場合は、担当者へおつなぎいたします。"
                    break

        return SupportResponse(
            answer=answer.strip(),
            needs_escalation=needs_escalation,
            sources=sources,
            chunk_ids=chunk_ids,
        )
