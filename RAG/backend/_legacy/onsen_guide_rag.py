"""
OnsenGuideRAG - 温泉ガイド向け推奨検索パイプライン
====================================================

検索戦略（温泉ガイド向け）:
  ① 初期検索: キーワード(BM25) + セマンティック（ハイブリッド）
  ② 候補抽出: 上位20件
  ③ 再ランク: CrossEncoderでスコアリング
  ④ 最終選択: スコア上位4件をLLMのコンテキストに使用
"""

import os
import json
from typing import List, Tuple
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.onsen_rag import OnsenRAG, DEFAULT_KUSATSU_CHUNKS_PATH

load_dotenv()


class OnsenGuideRAG(OnsenRAG):
    """
    温泉ガイド向けの推奨検索パイプラインを実装したRAG。

    OnsenRAGを継承し、load_json_chunks + query の流れで
    ハイブリッド検索 → 再ランク の4フェーズを実行する。
    """

    DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, cross_encoder_model: str = None, **kwargs):
        super().__init__(**kwargs)
        model_name = cross_encoder_model or self.DEFAULT_CROSS_ENCODER
        self.cross_encoder = CrossEncoder(model_name)
        self.documents: List[Document] = []  # BM25用

    def load_json_chunks(self, json_path: str | list[str] = None) -> None:
        """
        JSONチャンクを読み込み、ベクトルDBとBM25用ドキュメントリストを準備。
        """
        paths = (
            json_path
            if isinstance(json_path, list)
            else [json_path or DEFAULT_KUSATSU_CHUNKS_PATH]
        )
        all_chunks = []

        for file_path in paths:
            if not os.path.exists(file_path):
                print(f"⚠️ スキップ（未検出）: {file_path}")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise FileNotFoundError("読み込めるJSONチャンクファイルがありません。")

        def _to_str(val):
            if isinstance(val, list):
                return ",".join(str(v) for v in val)
            return str(val) if val is not None else ""

        documents = []
        for chunk in all_chunks:
            meta = chunk.get("metadata", {})
            tags_raw = meta.get("tags") or meta.get("keywords", [])
            tags_str = _to_str(tags_raw) if tags_raw else ""
            doc_metadata = {
                "chunk_id": chunk.get("chunk_id", ""),
                "source": meta.get("source", ""),
                "category": _to_str(meta.get("category", "")),
                "section": chunk.get("section", ""),
                "area": _to_str(meta.get("area", "")),
                "tags": tags_str,
            }
            doc = Document(
                page_content=chunk.get("content", ""),
                metadata=doc_metadata,
            )
            documents.append(doc)

        self.documents = documents

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

        print(f"✅ 温泉ガイド用データ準備完了: {len(documents)}件（ハイブリッド+再ランク対応）")

    def _rerank(self, query: str, docs: List[Document], top_k: int = 4) -> List[Tuple[Document, float]]:
        """CrossEncoderで再ランキング"""
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        ranked = list(zip(docs, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def query(
        self,
        question: str,
        initial_k: int = 20,
        final_k: int = 4,
        semantic_weight: float = 0.5,
    ) -> dict:
        """
        温泉ガイド向け4フェーズ検索で回答を生成。

        Args:
            question: 質問文
            initial_k: ①初期検索で取得する件数（デフォルト20）
            final_k: ④最終選択で採用する件数（デフォルト4）
            semantic_weight: ハイブリッド重み（0.5=キーワードとセマンティック同等）

        Returns:
            dict: "result"（回答）, "reranked_docs"（採用した文書とスコア）
        """
        if self.vectorstore is None:
            raise ValueError("データが未読み込みです。load_json_chunks()を実行してください。")

        if not self.documents:
            raise ValueError("BM25用ドキュメントがありません。load_json_chunks()で読み込んでください。")

        # ① 初期検索: キーワード + セマンティック（ハイブリッド）
        bm25 = BM25Retriever.from_documents(self.documents)
        bm25.k = initial_k
        vec_ret = self.vectorstore.as_retriever(search_kwargs={"k": initial_k})

        ensemble = EnsembleRetriever(
            retrievers=[vec_ret, bm25],
            weights=[semantic_weight, 1.0 - semantic_weight],
        )
        candidates = ensemble.invoke(question)

        # 重複除去（EnsembleRetrieverは同一ドキュメントを複数返す場合あり）
        seen = set()
        unique_candidates = []
        for doc in candidates:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_candidates.append(doc)

        # ② 候補を initial_k 件に制限
        candidates = unique_candidates[:initial_k]

        # ③ 再ランク: CrossEncoder
        reranked = self._rerank(question, candidates, top_k=final_k)

        # ④ 最終選択済み文書でコンテキスト作成
        context = "\n\n".join(doc.page_content for doc, _ in reranked)

        # プロンプト
        template = """
この情報の内容に基づいて質問に答えてください。
推測や記載のない情報は答えないでください。

【参考情報】
{context}

【質問】
{question}

【厳守事項】
・参考情報に含まれない事実や一般知識を使用しない
・推測や想像は行わない
・参考情報から判断できない場合は「参考情報からは分かりません」と回答する
・回答は簡潔で分かりやすい日本語にする

【回答形式】
・結論を最初に述べる
・必要に応じて箇条書きを使用
・根拠となる参考情報の要点を明示する

【回答】
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        formatted = prompt.format(context=context, question=question)
        response = self.llm.invoke(formatted)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "result": answer,
            "reranked_docs": reranked,
        }
