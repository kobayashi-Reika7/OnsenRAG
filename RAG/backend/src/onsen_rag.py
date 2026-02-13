"""
OnsenRAG - 温泉特化RAGシステム（多言語対応）
==============================================
温泉に関する質問に対してRAGで回答を生成する。

3段階検索パイプライン:
  質問 → ハイブリッド検索(initial_k件)
       → Re-ranking(CrossEncoder)
       → LLM候補抽出
       → スコア統合 → 上位k件
       → LLM回答生成

モジュール構成:
  config.py          - 定数・パス・パラメータ
  prompts.py         - プロンプトテンプレート
  data_loader.py     - データ読み込み
  search_pipeline.py - 検索パイプライン
  onsen_rag.py       - オーケストレーション（このファイル）
"""

import os
import re
import json
import hashlib
import time
from collections import OrderedDict

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_KUSATSU_CHUNKS_PATH,
    DEFAULT_JSON_CHUNK_PATHS,
    DEFAULT_TXT_PATHS,
    CHROMA_PERSIST_DIR,
    CHROMA_HASH_FILE,
    LOCATION_KEYWORDS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CROSS_ENCODER_MODEL,
    CONFIDENCE_THRESHOLD,
    QUERY_CACHE_MAXSIZE,
    QUERY_CACHE_TTL,
    DEFAULT_LANG,
    get_json_chunk_paths,
    get_chroma_persist_dir,
    get_chroma_hash_file,
    get_location_keywords,
    get_questions_path,
)
from src.prompts import ANSWER_PROMPT, load_prompt
from src.data_loader import load_json_chunks, load_txt_files
from src.text_splitter_utils import (
    create_token_text_splitter,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from src import search_pipeline


class OnsenRAG:
    """
    温泉情報に特化したRAGシステム（多言語対応）。

    使用例:
        rag = OnsenRAG(lang="ja")
        rag.load_from_data_folder()
        result = rag.query("草津温泉の特徴は？")
        print(result["result"])
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        semantic_weight: float = 0.5,
        initial_k: int = 10,
        final_k: int = 3,
        lang: str = DEFAULT_LANG,
    ):
        """
        Args:
            chunk_size: チャンクの最大トークン数
            chunk_overlap: チャンク間の重複トークン数
            semantic_weight: ハイブリッド検索でのセマンティック検索の重み（0.0〜1.0）
            initial_k: 初期検索で取得する件数
            final_k: 最終採用する件数
            lang: 言語コード ("ja" | "en")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self.initial_k = initial_k
        self.final_k = final_k
        self.lang = lang

        # 言語別設定
        self._location_keywords = get_location_keywords(lang)
        self._chroma_persist_dir = get_chroma_persist_dir(lang)
        self._chroma_hash_file = get_chroma_hash_file(lang)
        self._answer_prompt = load_prompt("answer.txt", lang)

        # Embedding モデル
        self.embeddings = HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDING_MODEL,
            model_kwargs={"local_files_only": True},
        )

        # CrossEncoder モデル（Re-ranking 用）
        print(f"[INIT] CrossEncoder loading... (lang={lang})")
        self.cross_encoder = CrossEncoder(
            DEFAULT_CROSS_ENCODER_MODEL,
            local_files_only=True,
        )
        print(f"  CrossEncoder: {DEFAULT_CROSS_ENCODER_MODEL}")

        # ベクトルストア
        self.vectorstore = None

        # 全ドキュメント（BM25 用）
        self.documents: list[Document] = []

        # BM25 キャッシュ
        self._bm25_all = None
        self._bm25_by_location: dict = {}

        # クエリキャッシュ
        self._query_cache: OrderedDict = OrderedDict()

        # 会話コンテキスト（直前の温泉地を保持）
        self._last_location: str | None = None

        # LLM
        from src.llm_factory import create_llm
        self.llm = create_llm(temperature=0)

    # ============================================================
    # データ読み込み
    # ============================================================

    def load_data(self, data_path: str = None) -> None:
        """テキストファイルを読み込みベクトルDBに保存（レガシー互換）。"""
        file_path = data_path or DEFAULT_DATA_PATH

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                "Please check data/onsen_knowledge.txt."
            )

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"[LOAD] Onsen data loaded: {len(text)} chars")

        document = Document(page_content=text)
        text_splitter = create_token_text_splitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splits = text_splitter.split_documents([document])
        print(f"[SPLIT] {len(splits)} chunks")

        self.documents = splits
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
        )
        print("[OK] Vector DB saved (hybrid search ready)")

    def load_json_chunks(self, json_path: str | list[str] = None) -> None:
        """JSON チャンクを読み込みベクトルDBに保存する。"""
        paths = (
            json_path
            if isinstance(json_path, list)
            else [json_path or DEFAULT_KUSATSU_CHUNKS_PATH]
        )

        documents = load_json_chunks(paths)
        self.documents = documents

        # ChromaDB 永続化
        data_hash = self._compute_data_hash(documents)
        cached = self._load_cached_vectorstore(data_hash)

        if cached:
            self.vectorstore = cached
            print("[CACHE HIT] ChromaDB loaded from disk")
        else:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self._chroma_persist_dir,
            )
            self._save_data_hash(data_hash)
            print("[OK] ChromaDB saved to disk")

        self._build_bm25_cache()
        print("[OK] Vector DB saved (hybrid search ready)")

    def load_from_data_folder(
        self,
        txt_paths: list[str] = None,
        json_paths: list[str] = None,
    ) -> None:
        """data フォルダ内の全データを統合して読み込む。"""
        txt_paths = txt_paths or DEFAULT_TXT_PATHS
        json_paths = json_paths or get_json_chunk_paths(self.lang)

        all_documents: list[Document] = []

        # テキストファイル読み込み
        if txt_paths:
            text_splitter = create_token_text_splitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            all_documents.extend(load_txt_files(txt_paths, text_splitter))

        # JSON チャンク読み込み
        try:
            all_documents.extend(load_json_chunks(json_paths))
        except FileNotFoundError:
            pass  # JSON がなくても txt だけで続行

        if not all_documents:
            raise FileNotFoundError(
                f"No data available. Please check the data folder ({DATA_DIR})."
            )

        self.documents = all_documents

        # ChromaDB 永続化
        data_hash = self._compute_data_hash(all_documents)
        cached = self._load_cached_vectorstore(data_hash)

        if cached:
            self.vectorstore = cached
            print(f"[CACHE HIT] ChromaDB loaded from disk ({len(all_documents)} chunks, lang={self.lang})")
        else:
            print(f"[BUILD] ChromaDB constructing ({len(all_documents)} chunks, lang={self.lang})...")
            self.vectorstore = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                persist_directory=self._chroma_persist_dir,
            )
            self._save_data_hash(data_hash)
            print(f"[OK] ChromaDB saved to {self._chroma_persist_dir}")

        self._build_bm25_cache()
        print(f"[OK] Total {len(all_documents)} chunks ready (hybrid search, lang={self.lang})")

    # ============================================================
    # ChromaDB キャッシュ
    # ============================================================

    @staticmethod
    def _compute_data_hash(documents: list[Document]) -> str:
        content = "".join(doc.page_content[:50] for doc in documents)
        content += str(len(documents))
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _load_cached_vectorstore(self, data_hash: str):
        if not os.path.exists(self._chroma_persist_dir):
            return None
        if not os.path.exists(self._chroma_hash_file):
            return None
        with open(self._chroma_hash_file, "r") as f:
            stored_hash = f.read().strip()
        if stored_hash != data_hash:
            print("[CACHE MISS] Data changed, rebuilding ChromaDB...")
            return None
        try:
            return Chroma(
                persist_directory=self._chroma_persist_dir,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            print(f"[CACHE ERROR] {e}, rebuilding...")
            return None

    def _save_data_hash(self, data_hash: str):
        os.makedirs(self._chroma_persist_dir, exist_ok=True)
        with open(self._chroma_hash_file, "w") as f:
            f.write(data_hash)

    # ============================================================
    # BM25 キャッシュ
    # ============================================================

    def _build_bm25_cache(self):
        """BM25 Retriever を事前構築してキャッシュする。"""
        if not self.documents:
            self._bm25_all = None
            self._bm25_by_location = {}
            return

        self._bm25_all = BM25Retriever.from_documents(self.documents)
        self._bm25_all.k = self.initial_k

        self._bm25_by_location = {}
        for loc_key in list(self._location_keywords.keys()) + ["onsen"]:
            loc_docs = [
                doc for doc in self.documents
                if doc.metadata.get("location") == loc_key
            ]
            if loc_docs:
                retriever = BM25Retriever.from_documents(loc_docs)
                retriever.k = self.initial_k
                self._bm25_by_location[loc_key] = retriever

        print(f"[BM25] Cached: all={len(self.documents)} docs, "
              f"locations={list(self._bm25_by_location.keys())}")

    # ============================================================
    # クエリキャッシュ
    # ============================================================

    def _cache_key(self, question: str, k: int) -> str:
        loc = self._last_location or ""
        return hashlib.md5(f"{question}::{k}::{loc}::{self.lang}".encode("utf-8")).hexdigest()

    def _get_from_cache(self, key: str) -> dict | None:
        if key not in self._query_cache:
            return None
        entry = self._query_cache[key]
        if time.time() - entry["timestamp"] > QUERY_CACHE_TTL:
            del self._query_cache[key]
            return None
        self._query_cache.move_to_end(key)
        print(f"[CACHE HIT] Query cache hit (TTL remaining: "
              f"{QUERY_CACHE_TTL - (time.time() - entry['timestamp']):.0f}s)")
        return entry["result"]

    def _put_to_cache(self, key: str, result: dict):
        if len(self._query_cache) >= QUERY_CACHE_MAXSIZE:
            self._query_cache.popitem(last=False)
        self._query_cache[key] = {
            "result": result,
            "timestamp": time.time(),
        }

    # ============================================================
    # 回答生成
    # ============================================================

    @staticmethod
    def _strip_chunk_ids(text: str) -> str:
        """LLM 回答から根拠・出典セクションを除去する。"""
        # ヘッダー以降を除去（日英両対応）
        text = re.split(
            r'\n*[#*_\-\s]*'
            r'(?:根拠\s*チャンク\s*I\s*D|根拠\s*チャンク|参照\s*チャンク\s*I\s*D|参照\s*チャンク'
            r'|参照\s*ソース|参照元|参考\s*情報|参考\s*文献|出典|引用元'
            r'|Sources?|References?|Citations?'
            r'|chunk_id|source|reference)'
            r'[*_\s]*[:：]?',
            text,
            flags=re.IGNORECASE,
        )[0]
        # インライン参照パターン (arima_001), [kusatsu_002] 等
        text = re.sub(
            r'[\(（\[]\s*(?:arima|kusatsu|hakone|beppu|onsen_knowledge)_\d+'
            r'(?:\s*[,、]\s*(?:arima|kusatsu|hakone|beppu|onsen_knowledge)_\d+)*'
            r'\s*[\)）\]]',
            '',
            text,
        )
        # 行内の chunk_id パターン（括弧なし）
        text = re.sub(
            r'(?:arima|kusatsu|hakone|beppu|onsen_knowledge)_\d{2,}',
            '',
            text,
        )
        # 「根拠チャンクID」等を含む行を丸ごと削除（日英両対応）
        text = re.sub(
            r'^.*(?:根拠チャンク|参照ソース|参照元|出典|Sources?|References?|Citations?|chunk_id).*$',
            '',
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    def query(self, question: str, k: int = 3) -> dict:
        """
        温泉に関する質問に対してRAGで回答を生成する。

        Returns:
            dict: {"result": 回答テキスト, "source_documents": Documentリスト, "chunk_ids": IDリスト}
        """
        if self.vectorstore is None:
            raise ValueError("Data not loaded. Please run load_data() first.")

        # クエリキャッシュ確認
        cache_key = self._cache_key(question, k)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # 3段階検索パイプライン
        docs, location = search_pipeline.hybrid_search(
            question,
            vectorstore=self.vectorstore,
            cross_encoder=self.cross_encoder,
            llm=self.llm,
            semantic_weight=self.semantic_weight,
            keyword_weight=self.keyword_weight,
            initial_k=self.initial_k,
            final_k=self.final_k,
            bm25_all=self._bm25_all,
            bm25_by_location=self._bm25_by_location,
            last_location=self._last_location,
            k=k,
            location_keywords=self._location_keywords,
        )

        # 会話コンテキスト更新
        if location:
            self._last_location = location

        # コンテキスト構築
        context_parts = []
        chunk_ids = []
        for doc in docs:
            cid = doc.metadata.get("chunk_id", "")
            if not cid and "source" in doc.metadata:
                cid = doc.metadata.get("source", "unknown").replace(".", "_")
            if not cid:
                cid = f"doc_{len(context_parts) + 1}"
            chunk_ids.append(cid)
            context_parts.append(doc.page_content)

        no_context = "(No relevant context found)" if self.lang == "en" else "（該当する文脈なし）"
        context = "\n\n---\n\n".join(context_parts) if context_parts else no_context

        # LLM 回答生成（429リトライ付き）
        from src.llm_factory import invoke_with_retry
        prompt = PromptTemplate(
            template=self._answer_prompt,
            input_variables=["context", "question"],
        )
        chain = prompt | self.llm
        answer = invoke_with_retry(chain, {"context": context, "question": question})
        answer = self._strip_chunk_ids(answer)

        result = {
            "result": answer.strip(),
            "source_documents": docs,
            "chunk_ids": chunk_ids,
        }

        self._put_to_cache(cache_key, result)
        return result

    # ============================================================
    # 評価
    # ============================================================

    def search_chunks(self, question: str, k: int = 3) -> list:
        """検索結果のみを返す（評価用・LLM 回答生成なし）。"""
        if self.vectorstore is None:
            raise ValueError("Data not loaded.")

        results, _ = search_pipeline.hybrid_search(
            question,
            vectorstore=self.vectorstore,
            cross_encoder=self.cross_encoder,
            llm=self.llm,
            semantic_weight=self.semantic_weight,
            keyword_weight=self.keyword_weight,
            initial_k=self.initial_k,
            final_k=self.final_k,
            bm25_all=self._bm25_all,
            bm25_by_location=self._bm25_by_location,
            last_location=self._last_location,
            k=k,
            location_keywords=self._location_keywords,
        )

        print(f"\n[SEARCH] Question: \"{question}\"")
        print(f"   Results: {len(results)} docs")
        for i, doc in enumerate(results):
            content = doc.page_content.replace("\n", " ")
            preview = content[:100] + "..." if len(content) > 100 else content
            safe_preview = preview.encode("ascii", errors="replace").decode()
            print(f"  [{i+1}] {safe_preview}")

        return results

    def evaluate(self, questions_path: str = None) -> list:
        """サンプル質問を使ってRAGの精度を一括評価する。"""
        if self.vectorstore is None:
            raise ValueError("Data not loaded.")

        file_path = questions_path or get_questions_path(self.lang)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        print("=" * 60)
        print("RAG Evaluation")
        print("=" * 60)

        for q in data["questions"]:
            question = q["question"]
            expected = q["expected_chunk"]
            keywords = q["expected_answer_keywords"]

            docs = self.search_chunks(question, k=3)

            all_text = " ".join([d.page_content for d in docs])
            matched_keywords = [kw for kw in keywords if kw in all_text]
            match_rate = len(matched_keywords) / len(keywords) if keywords else 0.0
            status = "[OK]" if match_rate >= 0.5 else "[WARN]" if match_rate > 0 else "[BAD]"

            print(f"   {status} Keyword match rate: {match_rate:.0%} "
                  f"({len(matched_keywords)}/{len(keywords)})")
            print(f"   Expected chunk: {expected}")
            print()

            results.append({
                "question": question,
                "expected_chunk": expected,
                "match_rate": match_rate,
                "matched_keywords": matched_keywords,
                "status": status,
            })

        total = len(results)
        good = sum(1 for r in results if r["status"] == "[OK]")
        warn = sum(1 for r in results if r["status"] == "[WARN]")
        bad = sum(1 for r in results if r["status"] == "[BAD]")

        print("=" * 60)
        print(f"Summary: OK={good} / WARN={warn} / BAD={bad} (total {total})")
        print("=" * 60)

        return results
