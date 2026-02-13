"""
ReRankingRAG - Re-rankingによる精度向上RAGシステム
=====================================================
初期検索結果をCrossEncoderでより正確に再スコアリングして上位のみ採用する。

なぜRe-rankingが有効か：
  - 初期検索（Bi-Encoder）は高速だが、精度がやや劣る
  - CrossEncoderはクエリと文書のペアを直接比較し、高精度にスコアリング
  - ただし計算コストが高いため、初期検索で絞った後に適用する

処理の流れ：
  1. 初期検索で多めに取得（initial_k=20）
  2. CrossEncoderで各文書をスコアリング
  3. スコア上位のみ採用（final_k=4）
  4. 採用した文書をコンテキストとしてLLMで回答生成
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from src.text_splitter_utils import create_token_text_splitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.documents import Document

# .envファイルから環境変数を読み込む
load_dotenv()


class ReRankingRAG:
    """
    Re-ranking（再ランキング）機能付きRAGシステム

    初期検索で多めに取得した文書をCrossEncoderで再スコアリングし、
    本当に関連度の高い文書だけを使って回答を生成する。

    CrossEncoder vs Bi-Encoder:
    - Bi-Encoder: クエリと文書を別々にベクトル化 → 高速だが精度はそこそこ
    - CrossEncoder: クエリと文書のペアを一緒に処理 → 低速だが高精度

    使用例:
        rag = ReRankingRAG()
        rag.load_documents(["テキスト1", "テキスト2"])
        result = rag.query("質問内容")
    """

    # CrossEncoderモデル名（軽量で高速なモデル）
    DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, cross_encoder_model: str = None):
        """
        ReRankingRAGの初期化

        Args:
            cross_encoder_model: CrossEncoderモデル名
                                デフォルトは ms-marco-MiniLM-L-6-v2（軽量・高速）
        """
        # CrossEncoderモデルの初期化
        # クエリと文書のペアに対して関連度スコアを算出する
        model_name = cross_encoder_model or self.DEFAULT_CROSS_ENCODER
        self.cross_encoder = CrossEncoder(model_name)

        # Embeddingモデル（日本語対応）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        # ベクトルストア
        self.vectorstore = None

        # LLMの初期化
        # temperature=0: ハルシネーション（幻覚的回答）を抑制
        self.llm = OpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def load_documents(self, texts: list[str]) -> None:
        """
        テキストデータをベクトルDBに読み込む

        Args:
            texts: 読み込むテキストのリスト
        """
        documents = [Document(page_content=text) for text in texts]

        # テキストをトークンベースで分割
        text_splitter = create_token_text_splitter(
            document_type="general",
            separators=["\n\n", "\n", "。", "、", " ", ""],
        )
        splits = text_splitter.split_documents(documents)

        # ベクトルDBに保存
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

        print(f"✅ {len(splits)}件のチャンクをベクトルDBに保存しました")

    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        CrossEncoderで文書を再ランキングする

        処理の流れ：
        1. クエリと各文書のペアを作成
        2. CrossEncoderでペアごとにスコアを計算
        3. スコアの高い順にソート
        4. 上位top_k件のみ返す

        Args:
            query: 検索クエリ（質問文）
            documents: 初期検索で取得した文書のリスト
            top_k: 返す文書の最大数

        Returns:
            List[Tuple[Document, float]]: (文書, スコア)のリスト（スコア降順）
        """
        # クエリと各文書のペアを作成
        # CrossEncoderは2つのテキストの関連度を直接計算する
        pairs = [[query, doc.page_content] for doc in documents]

        # CrossEncoderでスコアリング
        # 各ペアに対して0〜1の関連度スコアを算出
        scores = self.cross_encoder.predict(pairs)

        # スコアと文書を紐付け
        document_score_pairs = list(zip(documents, scores))

        # スコアの降順でソート（関連度が高い順）
        document_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 上位top_k件を返す
        return document_score_pairs[:top_k]

    def query(
        self,
        question: str,
        initial_k: int = 20,
        final_k: int = 4
    ) -> dict:
        """
        Re-ranking付きで回答を生成

        処理の流れ：
        1. 初期検索: initial_k件を取得（多めに取得）
        2. Re-ranking: CrossEncoderでスコアリング
        3. 上位選択: final_k件に絞る
        4. 回答生成: 絞った文書をコンテキストとしてLLMで回答

        Args:
            question: 質問文
            initial_k: 初期検索で取得する件数（デフォルト20）
                      多めに取ってRe-rankingで絞る戦略
            final_k: Re-ranking後に採用する件数（デフォルト4）

        Returns:
            dict: 回答結果
                - "result": LLMが生成した回答テキスト
                - "reranked_docs": Re-ranking後の文書とスコア
        """
        # ベクトルストアが未作成の場合はエラー
        if self.vectorstore is None:
            raise ValueError(
                "データが未読み込みです。"
                "先にload_documents()でデータを読み込んでください。"
            )

        # ステップ1: 初期検索（多めに取得）
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": initial_k}
        )
        initial_docs = retriever.invoke(question)
        print(f"🔍 初期検索: {len(initial_docs)}件取得")

        # ステップ2: Re-ranking（CrossEncoderでスコアリング）
        reranked = self.rerank_documents(
            query=question,
            documents=initial_docs,
            top_k=final_k
        )

        # Re-ranking結果を表示
        print(f"🏆 Re-ranking後: 上位{len(reranked)}件を採用")
        for i, (doc, score) in enumerate(reranked):
            preview = doc.page_content[:80] + "..." \
                if len(doc.page_content) > 80 else doc.page_content
            print(f"  [{i+1}] スコア: {score:.4f} | {preview}")

        # ステップ3: Re-ranking後の文書でコンテキストを作成
        context = "\n\n".join([doc.page_content for doc, _ in reranked])

        # ステップ4: 日本語対応プロンプトで回答生成
        template = """あなたは専門的なアシスタントです。以下の文脈のみを使用して、質問に日本語で正確に答えてください。

【文脈】
{context}

【質問】
{question}

【回答の際の注意点】
・文脈に書かれている事実のみを使用する
・推測や一般知識を混ぜない
・答えられない場合は正直にその旨を伝える

【回答】"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # LLMで回答生成
        formatted_prompt = prompt.format(
            context=context,
            question=question
        )
        answer = self.llm.invoke(formatted_prompt)

        return {
            "result": answer,
            "reranked_docs": reranked
        }
