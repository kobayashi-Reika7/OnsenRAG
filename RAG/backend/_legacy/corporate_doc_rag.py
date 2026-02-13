"""
CorporateDocRAG - 社内ドキュメント検索システム
================================================
フォルダ内のPDF・TXTを一括読み込みし、社内文書への質問にRAGで回答する。

機能:
- 複数ファイル（PDF, TXT）の一括読み込み
- メタデータ（ソースファイル名）の保持
- 社内文書向けプロンプト
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.text_splitter_utils import create_token_text_splitter

load_dotenv()


def _load_llm():
    """Gemini → Groq → OpenAI の順でフォールバック"""
    import os
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


class CorporateDocRAG:
    """
    社内ドキュメント検索RAGシステム

    フォルダ内のPDF・TXTを読み込み、社内文書に関する質問に回答する。
    """

    def __init__(self, persist_directory: str = "./chroma_corporate_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )
        self.vectorstore = None
        self.llm = _load_llm()

    def load_from_folder(
        self,
        folder_path: str,
        document_type: str = "general",
        extensions: tuple = (".pdf", ".txt"),
    ) -> None:
        """
        フォルダ内のドキュメントを一括読み込み

        Args:
            folder_path: ドキュメントフォルダのパス
            document_type: チャンキングプリセット
            extensions: 読み込む拡張子（.pdf, .txt）
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"フォルダが見つかりません: {folder_path}")

        all_docs = []
        for ext in extensions:
            for file_path in folder.glob(f"**/*{ext}"):
                if file_path.is_file():
                    docs = self._load_file(str(file_path))
                    for doc in docs:
                        doc.metadata["source_file"] = file_path.name
                    all_docs.extend(docs)

        if not all_docs:
            raise ValueError(f"読み込めるドキュメントがありません: {folder_path}")

        text_splitter = create_token_text_splitter(document_type=document_type)
        splits = text_splitter.split_documents(all_docs)

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"[OK] 社内ドキュメント {len(splits)} チャンクを登録しました")

    def _load_file(self, path: str) -> list[Document]:
        """単一ファイルを読み込み"""
        path_lower = path.lower()
        if path_lower.endswith(".pdf"):
            loader = PyPDFLoader(path)
            return loader.load()
        if path_lower.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": path})]
        return []

    def query(self, question: str, k: int = 4) -> dict:
        """社内文書に対して質問"""
        if self.vectorstore is None:
            raise ValueError("先に load_from_folder() を実行してください。")

        template = """あなたは社内ドキュメント検索アシスタントです。
以下の文脈（社内文書の抜粋）のみを使用して、質問に正確に答えてください。

【文脈】
{context}

【質問】
{question}

【回答の際の注意点】
・文脈に書かれている事実のみを使用する
・推測や一般知識を混ぜない
・答えられない場合は「文書内に該当情報が見つかりません」と回答する
・回答は簡潔に、必要に応じて出典（ソースファイル）を明示する

【回答】
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": prompt},
        )
        out = qa_chain({"query": question})
        if hasattr(out.get("result"), "content"):
            out["result"] = out["result"].content
        return out
