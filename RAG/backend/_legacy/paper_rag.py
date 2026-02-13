"""
PaperRAG - 研究論文アシスタント
================================
PDF論文を読み込み、引用形式で回答するRAGシステム。

機能:
- 論文PDFの読み込み（long プリセット: 1000 tokens, overlap 150）
- 引用形式の回答（[1], [2] などの出典番号）
- アブストラクト・要約の取得
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.text_splitter_utils import create_token_text_splitter

load_dotenv()


def _load_llm():
    """Gemini → Groq → OpenAI の順でフォールバック"""
    google_key = os.getenv("GOOGLE_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if google_key and not google_key.startswith("your_"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        except Exception:
            pass
    if groq_key and not groq_key.startswith("gsk_your"):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        except Exception:
            pass
    if openai_key and not openai_key.startswith("sk-your"):
        try:
            from langchain_openai import OpenAI
            return OpenAI(temperature=0)
        except Exception:
            pass
    raise ValueError(".env に API キーを設定してください。")


class PaperRAG:
    """
    研究論文アシスタントRAG

    論文PDFを読み込み、引用形式で回答を生成する。
    """

    def __init__(self, persist_directory: str = "./chroma_paper_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )
        self.vectorstore = None
        self.docs_with_index: list[tuple[int, Document]] = []  # 引用番号用
        self.llm = _load_llm()

    def load_paper(self, pdf_path: str) -> None:
        """
        論文PDFを読み込み

        Args:
            pdf_path: PDFファイルのパス
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDFが見つかりません: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # 論文向け: long プリセット（1000 tokens, overlap 150）
        text_splitter = create_token_text_splitter(document_type="long")
        splits = text_splitter.split_documents(pages)

        # メタデータにページ番号を付与
        for i, doc in enumerate(splits):
            doc.metadata["chunk_index"] = i + 1
            doc.metadata["source"] = Path(pdf_path).name

        self.docs_with_index = [(i + 1, doc) for i, doc in enumerate(splits)]
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"✅ 論文を読み込みました: {len(splits)} チャンク")

    def query(self, question: str, k: int = 4) -> dict:
        """
        論文に対して質問し、引用形式で回答

        Returns:
            dict: {"result": 回答, "sources": [出典リスト]}
        """
        if self.vectorstore is None:
            raise ValueError("先に load_paper() を実行してください。")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved = retriever.invoke(question)

        # コンテキストを [1], [2] 形式で構築
        context_parts = []
        source_list = []
        for i, doc in enumerate(retrieved):
            idx = i + 1
            source_list.append(f"[{idx}] {doc.metadata.get('source', '')} p.{doc.metadata.get('page', '?')}")
            context_parts.append(f"[{idx}] {doc.page_content}")

        context = "\n\n".join(context_parts)

        template = """あなたは研究論文アシスタントです。
以下の文脈は論文の抜粋で、[1][2]は出典番号です。
文脈のみを使用して質問に答えてください。

【文脈】
{context}

【質問】
{question}

【回答の際の注意点】
・文脈に書かれている事実のみを使用する
・回答中で根拠となる部分に [1], [2] のように出典番号を付ける
・推測や一般知識を混ぜない
・答えられない場合は「論文内に該当情報が見つかりません」と回答する
・回答は簡潔な日本語で、学術的な表現を心がける

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
            "sources": source_list,
            "source_documents": retrieved,
        }

    def get_abstract(self, pdf_path: str = None) -> str:
        """
        論文の最初のページ（アブストラクト付近）を取得

        load_paper 済みの場合、最初のチャンクを返す。
        """
        if self.vectorstore is None and pdf_path:
            self.load_paper(pdf_path)

        if not self.docs_with_index:
            return "論文が読み込まれていません。"

        # 最初のチャンク（通常アブストラクトを含む）
        _, first_doc = self.docs_with_index[0]
        return first_doc.page_content[:1500]
