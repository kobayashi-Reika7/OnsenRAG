"""
PDFRagSystem - PDF対応RAGシステム
===================================
PDFファイルからRAGシステムを構築するクラス。
日本語文書に最適化されたEmbeddingモデルとプロンプトを使用。

処理フロー：
  1. PDFロード → PyPDFLoaderでPDFを読み込み
  2. チャンク分割 → 日本語に最適化した分割方法
  3. ベクトル化 → 日本語対応Embeddingモデル使用
  4. 回答生成 → カスタムプロンプトでLLM実行
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from src.text_splitter_utils import create_token_text_splitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# .envファイルから環境変数を読み込む
load_dotenv()


class PDFRagSystem:
    """
    PDF文書対応のRAGシステム（日本語最適化）

    特徴：
    - multilingual-e5-base Embeddingモデルで日本語対応
    - 日本語の句読点（。、）を考慮したテキスト分割
    - 日本語応答のカスタムプロンプト
    - ベクトルDBのディスク永続化対応

    使用例:
        rag = PDFRagSystem()
        rag.load_pdf("document.pdf")
        result = rag.query("この文書の要点は何ですか？")
        print(result["result"])
    """

    # ベクトルDBの保存先ディレクトリ（デフォルト値）
    DEFAULT_PERSIST_DIR = "./chroma_pdf_db"

    def __init__(self, persist_directory: str = None):
        """
        PDFRagSystemの初期化

        Args:
            persist_directory: ベクトルDBの保存先パス
                              指定しない場合は ./chroma_pdf_db に保存

        Embeddingモデル:
          - multilingual-e5-base を使用（日本語対応）
          - 英語のみのモデルと比べて日本語の精度が大幅に向上

        LLM:
          - temperature=0.3 で少しだけ多様性のある回答を許容
        """
        # 保存先ディレクトリ（指定がなければデフォルト値を使用）
        self.persist_directory = persist_directory or self.DEFAULT_PERSIST_DIR

        # 日本語対応Embeddingモデル
        # multilingual-e5-base: 多言語に対応した高精度モデル
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        # ベクトルストア（初期状態はNone）
        self.vectorstore = None

        # LLMの初期化
        # temperature=0.3: やや創造的な回答を許容
        self.llm = OpenAI(
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def load_pdf(self, pdf_path: str) -> None:
        """
        PDFファイルを読み込み、ベクトル化してDBに保存

        処理の流れ：
        1. PyPDFLoaderでPDFをページ単位で読み込み
        2. 日本語に適したセパレータでテキストを分割
        3. ベクトル化してChromaDBに保存（ディスクに永続化）

        Args:
            pdf_path: PDFファイルのパス
                     例: "./data/onsen_guide.pdf"

        ポイント:
          - document_type="general": 一般的な文書向けプリセット（600 tokens, overlap 75）
          - separators: 日本語の句読点（。、）を分割ポイントに指定
        """
        # PDFファイルの存在チェック
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"PDFファイルが見つかりません: {pdf_path}"
            )

        # PDFをページ単位で読み込み
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"📄 PDFを読み込みました: {len(pages)}ページ")

        # 日本語に適したテキスト分割設定
        # separators: 分割の優先順位（左から順に試行）
        #   "\n\n" → 段落区切り（最優先）
        #   "\n"   → 改行
        #   "。"   → 日本語の文末
        #   "、"   → 日本語の読点
        #   " "    → スペース
        #   ""     → 文字単位（最終手段）
        text_splitter = create_token_text_splitter(
            document_type="general",
            separators=["\n\n", "\n", "。", "、", " ", ""],
        )
        splits = text_splitter.split_documents(pages)
        print(f"✂️ {len(splits)}件のチャンクに分割しました")

        # ベクトルDB作成（ディスクに永続化）
        # persist_directory を指定することで、次回起動時に再利用可能
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        print(f"✅ ベクトルDBに保存しました: {self.persist_directory}")

    def load_existing_db(self) -> None:
        """
        既存のベクトルDBを読み込む（再起動時に使用）

        一度load_pdfで作成したDBを再利用する場合に使用。
        毎回PDFを読み込む必要がなくなり、起動が高速化する。
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"ベクトルDBが見つかりません: {self.persist_directory}\n"
                "先にload_pdf()でPDFを読み込んでください。"
            )

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print(f"✅ 既存のベクトルDBを読み込みました: {self.persist_directory}")

    def query(self, question: str, k: int = 4) -> dict:
        """
        質問に対してRAGで日本語回答を生成

        日本語対応のカスタムプロンプトを使用して、
        検索された文脈のみに基づいた正確な回答を生成する。

        Args:
            question: 質問文（日本語）
                     例: "この文書の要点は何ですか？"
            k: 検索結果の件数（デフォルト4件）

        Returns:
            dict: 回答結果
                - "result": LLMが生成した回答テキスト
        """
        # ベクトルストアが未作成の場合はエラー
        if self.vectorstore is None:
            raise ValueError(
                "ベクトルストアが未作成です。"
                "先にload_pdf()またはload_existing_db()を実行してください。"
            )

        # 日本語対応のカスタムプロンプト
        # 文脈のみを使用し、推測を避けるよう指示
        template = """あなたは専門的なアシスタントです。以下の文脈のみを使用して、質問に日本語で正確に答えてください。

【文脈】
{context}

【質問】
{question}

【回答の際の注意点】
・文脈に書かれている事実のみを使用する
・推測や一般知識を混ぜない
・答えられない場合は正直にその旨を伝える
・丁寧で分かりやすい日本語で回答する

【回答】"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # QAチェーン作成と実行
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            chain_type_kwargs={"prompt": prompt}
        )

        result = qa_chain({"query": question})
        return result
