"""
HybridSearchRAG - ハイブリッド検索RAGシステム
================================================
2つの異なる検索手法を組み合わせて精度を向上させる：

  セマンティック検索（ベクトル類似度）
    → 意味的な類似性で検索（「犬」で検索→「ペット」もヒット）

  + キーワード検索（BM25）
    → 単語の一致度で検索（「犬」で検索→「犬」を含む文書がヒット）

  = ハイブリッド検索（EnsembleRetriever）
    → 両方の結果を重み付けで統合し、より精度の高い検索を実現
"""

import os
from dotenv import load_dotenv
from src.text_splitter_utils import create_token_text_splitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.documents import Document

# .envファイルから環境変数を読み込む
load_dotenv()


class HybridSearchRAG:
    """
    ハイブリッド検索RAGシステム

    セマンティック検索（ベクトル類似度）とキーワード検索（BM25）を
    組み合わせることで、検索精度を向上させる。

    なぜハイブリッドが有効か：
    - セマンティック検索: 意味は理解するが、固有名詞に弱い
    - キーワード検索: 固有名詞に強いが、類義語に弱い
    - 両方を組み合わせると、互いの弱点を補完できる

    使用例:
        rag = HybridSearchRAG()
        rag.load_documents(["テキスト1", "テキスト2"])
        result = rag.query("質問内容")
    """

    def __init__(self, semantic_weight: float = 0.5):
        """
        HybridSearchRAGの初期化

        Args:
            semantic_weight: セマンティック検索の重み（0.0〜1.0）
                            デフォルト0.5（セマンティックとキーワードを同等に扱う）
                            0.7にするとセマンティック検索を重視
                            0.3にするとキーワード検索を重視
        """
        # セマンティック検索とキーワード検索の重み配分
        self.semantic_weight = semantic_weight
        # キーワード検索の重みは残りの値
        self.keyword_weight = 1.0 - semantic_weight

        # Embeddingモデル（日本語対応）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        # ベクトルストアとドキュメントリスト
        self.vectorstore = None
        self.documents = []  # BM25 Retriever用にドキュメントを保持

        # LLMの初期化
        self.llm = OpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def load_documents(self, texts: list[str]) -> None:
        """
        テキストデータを読み込み、ハイブリッド検索用に準備

        処理の流れ：
        1. テキストをDocumentオブジェクトに変換
        2. チャンク化して分割
        3. ベクトルDB（Chroma）に保存 → セマンティック検索用
        4. ドキュメントリストを保持 → BM25キーワード検索用

        Args:
            texts: 読み込むテキストのリスト
        """
        # テキストをDocumentオブジェクトに変換
        documents = [Document(page_content=text) for text in texts]

        # テキストをトークンベースで分割
        text_splitter = create_token_text_splitter(
            document_type="general",
            separators=["\n\n", "\n", "。", "、", " ", ""],
        )
        splits = text_splitter.split_documents(documents)

        # 分割後のドキュメントを保持（BM25 Retriever用）
        self.documents = splits

        # ベクトルDBに保存（セマンティック検索用）
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

        print(f"✅ {len(splits)}件のチャンクを読み込みました（ハイブリッド検索対応）")

    def _create_ensemble_retriever(self, k: int = 4) -> EnsembleRetriever:
        """
        セマンティック検索とキーワード検索を統合したRetrieverを作成

        Args:
            k: 各検索手法が返す結果の件数

        Returns:
            EnsembleRetriever: 統合されたRetriever

        内部処理：
        - BM25Retriever: キーワードの出現頻度で関連度を計算
        - VectorStoreRetriever: ベクトルの距離（コサイン類似度）で関連度を計算
        - EnsembleRetriever: 両方の結果を重み付けで統合
        """
        # キーワード検索用のBM25 Retriever
        # BM25: テキスト中の単語の出現頻度に基づくランキングアルゴリズム
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = k  # 上位k件を取得

        # ベクトル検索用のRetriever
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # 両方を組み合わせたEnsemble Retriever
        # weights: セマンティック検索とキーワード検索の重み配分
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[self.semantic_weight, self.keyword_weight]
        )

        return ensemble_retriever

    def query(self, question: str, k: int = 4) -> dict:
        """
        ハイブリッド検索で回答を生成

        セマンティック検索とキーワード検索の両方を使って
        関連するドキュメントを取得し、LLMで回答を生成する。

        Args:
            question: 質問文
            k: 検索結果の件数（デフォルト4件）

        Returns:
            dict: 回答結果
                - "result": LLMが生成した回答テキスト
        """
        # ベクトルストアが未作成の場合はエラー
        if self.vectorstore is None:
            raise ValueError(
                "データが未読み込みです。"
                "先にload_documents()でデータを読み込んでください。"
            )

        # ハイブリッドRetrieverを作成
        ensemble_retriever = self._create_ensemble_retriever(k=k)

        # 日本語対応プロンプト
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

        # QAチェーン作成
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            chain_type_kwargs={"prompt": prompt}
        )

        result = qa_chain({"query": question})
        return result

    def compare_search_modes(
        self, question: str, k: int = 4
    ) -> dict[str, list[Document]]:
        """
        キーワード・セマンティック・ハイブリッドの3手法で検索し、回収率を比較する。

        精度の測定: 各手法の結果を比較して、どれが最適か検証するヒント。

        Args:
            question: 検索クエリ
            k: 各手法の取得件数

        Returns:
            dict: {"keyword": [...], "semantic": [...], "hybrid": [...]}
        """
        if self.vectorstore is None:
            raise ValueError("データが未読み込みです。")

        # キーワード検索（BM25）のみ
        bm25 = BM25Retriever.from_documents(self.documents)
        bm25.k = k
        keyword_docs = bm25.invoke(question)

        # セマンティック検索（ベクトル）のみ
        vec_ret = self.vectorstore.as_retriever(search_kwargs={"k": k})
        semantic_docs = vec_ret.invoke(question)

        # ハイブリッド検索
        ensemble = self._create_ensemble_retriever(k=k)
        hybrid_docs = ensemble.invoke(question)

        return {
            "keyword": keyword_docs,
            "semantic": semantic_docs,
            "hybrid": hybrid_docs,
        }

    def search_only(self, question: str, k: int = 4) -> list:
        """
        回答生成なしで、検索結果のみを取得（デバッグ・検証用）

        検索精度を確認したい時に使用。
        セマンティック検索とキーワード検索それぞれの結果を比較可能。

        Args:
            question: 検索クエリ
            k: 取得件数

        Returns:
            list: 検索結果のDocumentリスト
        """
        if self.vectorstore is None:
            raise ValueError("データが未読み込みです。")

        ensemble_retriever = self._create_ensemble_retriever(k=k)
        results = ensemble_retriever.invoke(question)

        # 検索結果を見やすく表示
        print(f"🔍 「{question}」の検索結果: {len(results)}件")
        for i, doc in enumerate(results):
            # 長いテキストは100文字で切り詰めて表示
            preview = doc.page_content[:100] + "..." \
                if len(doc.page_content) > 100 else doc.page_content
            print(f"  [{i+1}] {preview}")

        return results
