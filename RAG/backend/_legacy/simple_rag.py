"""
SimpleRAG - 基本的なRAGシステム
================================
テキストデータを入力し、ベクトル化してChromaDBに保存。
質問に対して関連するテキストを検索し、LLMで回答を生成する。

処理の流れ：
  ステップ1: テキストをチャンク化（分割）
  ステップ2: ベクトル化（Embedding）
  ステップ3: Chromaにベクトルを格納
  ステップ4: RetrievalQAで回答生成
"""

import os
from dotenv import load_dotenv
from src.text_splitter_utils import create_token_text_splitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.documents import Document

# .envファイルから環境変数を読み込む（APIキーなど）
load_dotenv()


class SimpleRAG:
    """
    基本的なRAG（Retrieval-Augmented Generation）システム

    テキストデータを受け取り、以下の処理を行う：
    1. テキストを適切なサイズに分割（チャンク化）
    2. 各チャンクをベクトル（数値の配列）に変換
    3. ベクトルをChromaDBに保存
    4. 質問に対して類似するチャンクを検索し、LLMで回答生成

    使用例:
        rag = SimpleRAG()
        rag.load_documents(["テキスト1", "テキスト2"])
        result = rag.query("質問内容")
    """

    def __init__(self):
        """
        SimpleRAGの初期化

        Embeddingモデル:
          - all-MiniLM-L6-v2を使用（無料・ローカルで動作）
          - テキストを384次元のベクトルに変換する軽量モデル

        LLM:
          - OpenAIのGPTモデルを使用
          - temperature=0 で決定論的な回答を生成（毎回同じ入力に同じ出力）
        """
        # Embeddingモデル（無料・ローカル動作）
        # テキストを数値ベクトルに変換するために使用
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ベクトルストア（初期状態はNone、load_documentsで作成）
        self.vectorstore = None

        # LLM（大規模言語モデル）の初期化
        # temperature=0: ランダム性なし → 安定した回答
        self.llm = OpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def load_documents(self, texts: list[str]) -> None:
        """
        テキストデータをベクトルDBに読み込む

        処理の流れ：
        1. テキストをDocumentオブジェクトに変換
        2. RecursiveCharacterTextSplitterでチャンク化
        3. チャンクをベクトル化してChromaDBに保存

        Args:
            texts: 読み込むテキストのリスト
                   例: ["温泉の歴史は...", "効能について..."]
        """
        # テキストをLangChainのDocumentオブジェクトに変換
        # Documentはpage_content（本文）とmetadata（付随情報）を持つ
        documents = [Document(page_content=text) for text in texts]

        # テキストをトークンベースで分割（general プリセット: 600 tokens, overlap 75）
        text_splitter = create_token_text_splitter(document_type="general")
        splits = text_splitter.split_documents(documents)

        # ベクトルDBに保存
        # 各チャンクをEmbeddingモデルでベクトル化し、ChromaDBに格納
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

        print(f"✅ {len(splits)}件のチャンクをベクトルDBに保存しました")

    def query(self, question: str, k: int = 3) -> dict:
        """
        質問に対してRAGで回答を生成する

        処理の流れ：
        1. 質問文をベクトル化
        2. ChromaDBから類似するチャンクをk件検索
        3. 検索結果をコンテキストとしてLLMに渡す
        4. LLMが回答を生成

        Args:
            question: 質問文
                     例: "温泉の効能は何ですか？"
            k: 検索結果の件数（デフォルト3件）
               多すぎるとノイズが増え、少なすぎると情報不足になる

        Returns:
            dict: 回答結果
                - "result": LLMが生成した回答テキスト
                - "source_documents": 参照した元のチャンク
        """
        # ベクトルストアが未作成の場合はエラー
        if self.vectorstore is None:
            raise ValueError(
                "ベクトルストアが未作成です。"
                "先にload_documents()でデータを読み込んでください。"
            )

        # RetrievalQAチェーンで回答を生成
        # chain_type="stuff": 検索結果をすべてプロンプトに含める方式
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            return_source_documents=True
        )

        # 質問を実行して回答を取得
        result = qa_chain({"query": question})
        return result
