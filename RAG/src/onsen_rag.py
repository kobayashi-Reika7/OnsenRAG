"""
OnsenRAG - 温泉特化RAGシステム
==================================
温泉テキストデータ（data/onsen_knowledge.txt）を読み込み、
温泉に関する質問に対してRAGで回答を生成する。

このクラスの特徴：
- 温泉テキストを「■」見出し単位で意味的に分割
- 日本語に最適化されたEmbeddingモデルとプロンプト
- 評価用の検索結果取得メソッド付き

RAG教材としてのポイント：
- 情報が分散している → 検索が効く
- 条件付き質問が多い → RAG必須
- 嘘をつくとすぐ分かる → 精度評価しやすい
"""

import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from src.text_splitter_utils import (
    create_token_text_splitter,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# .envファイルから環境変数を読み込む
load_dotenv()

# data フォルダのパス
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# データファイルのパス（プロジェクトルートからの相対パス）
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "onsen_knowledge.txt")

# サンプル質問ファイルのパス
DEFAULT_QUESTIONS_PATH = os.path.join(DATA_DIR, "sample_questions.json")

# JSONチャンクファイルのパス（草津温泉ガイド等の構造化データ）
DEFAULT_KUSATSU_CHUNKS_PATH = os.path.join(DATA_DIR, "kusatsu_chunks.json")

# 場所別統合チャンクファイル + 温泉基礎知識
DEFAULT_JSON_CHUNK_PATHS = [
    os.path.join(DATA_DIR, "kusatsu_chunks.json"),        # 草津温泉（104 chunks）
    os.path.join(DATA_DIR, "hakone_chunks.json"),          # 箱根温泉（45 chunks）
    os.path.join(DATA_DIR, "beppu_chunks.json"),           # 別府温泉（20 chunks）
    os.path.join(DATA_DIR, "arima_chunks.json"),           # 有馬温泉（19 chunks）
    os.path.join(DATA_DIR, "onsen_knowledge_chunks.json"), # 温泉基礎知識（11 chunks）
]

# テキストファイルは全て JSON チャンクに変換済みのため空
DEFAULT_TXT_PATHS = []


# 温泉地名 → chunk_idプレフィックスの対応表
# 質問文に含まれるキーワードで、どの温泉地のチャンクを検索するか判定する
LOCATION_KEYWORDS = {
    "kusatsu": ["草津"],
    "hakone": ["箱根"],
    "beppu": ["別府"],
    "arima": ["有馬"],
}


class OnsenRAG:
    """
    温泉情報に特化したRAGシステム

    温泉テキストデータを「■」見出し単位で意味的に分割し、
    質問に対して関連するチャンクを検索してLLMで回答を生成する。

    なぜ「■」で分割するのか：
    - テキストが見出し（■）ごとにまとまった意味を持つ
    - 機械的な文字数分割より、意味単位の分割のほうが精度が高い
    - 「冬におすすめの温泉地は？」→「■ 季節ごとの楽しみ方」がヒットしやすい

    使用例:
        rag = OnsenRAG()
        rag.load_data()
        result = rag.query("草津温泉の特徴は？")
        print(result["result"])
    """

    # CrossEncoderモデル名（軽量で高速なモデル）
    # クエリと文書のペアを直接比較し、関連度を高精度に計算する
    DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        semantic_weight: float = 0.5,
        initial_k: int = 10,
        final_k: int = 3,
    ):
        """
        OnsenRAGの初期化

        Args:
            chunk_size: チャンクの最大トークン数（デフォルト600、general プリセット）
            chunk_overlap: チャンク間の重複トークン数（デフォルト75）
            semantic_weight: ハイブリッド検索でのセマンティック検索の重み（0.0〜1.0）
                            デフォルト0.5（セマンティックとキーワードを同等に扱う）
                            0.7 → セマンティック重視、0.3 → キーワード重視
            initial_k: 初期検索で取得する件数（多めに取ってRe-rankingで絞る）
            final_k: Re-ranking後に最終採用する件数

        検索パイプライン:
          質問 → ハイブリッド検索(initial_k件) → Re-ranking(CrossEncoder) → 上位final_k件 → LLM回答

        トークンベース管理の理由：
        - LLMのコンテキスト制限はトークン数で表現される
        - 文字数より正確なチャンクサイズ制御が可能
        - 日本語は1文字≒2〜3トークン程度のため、トークン単位が適切
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # ハイブリッド検索の重み（セマンティック vs キーワード）
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight

        # Re-ranking用の初期検索件数 / 最終採用件数
        self.initial_k = initial_k
        self.final_k = final_k

        # 日本語対応Embeddingモデル
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        # CrossEncoderモデル（Re-ranking用）
        # Bi-Encoderより低速だが、クエリと文書のペアを直接比較するため高精度
        print("[INIT] CrossEncoder loading...")
        self.cross_encoder = CrossEncoder(self.DEFAULT_CROSS_ENCODER)
        print(f"  CrossEncoder: {self.DEFAULT_CROSS_ENCODER}")

        # ベクトルストア（セマンティック検索用）
        self.vectorstore = None

        # 全ドキュメントリスト（BM25キーワード検索用に保持）
        self.documents = []

        # LLMの初期化（Gemini優先 → Groq → OpenAI）
        self.llm = self._init_llm()

    def _init_llm(self):
        """
        LLMを初期化する。
        Google API キーがあれば Gemini（無料枠あり・高性能）を使用し、
        なければ Groq → OpenAI の順にフォールバックする。

        Geminiの利点:
        - 無料枠あり（1分あたり15リクエスト）
        - 日本語性能が高い
        - gemini-2.0-flash は高速かつ高精度
        """
        google_key = os.getenv("GOOGLE_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")

        if google_key and not google_key.startswith("your_"):
            try:
                import concurrent.futures
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0,
                    google_api_key=google_key,
                    max_retries=1,
                )
                # 接続テスト（タイムアウト10秒でクォータ切れを早期検出）
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(llm.invoke, "test")
                    future.result(timeout=10)
                print("  LLM: Gemini (gemini-2.0-flash)")
                return llm
            except Exception as e:
                safe_msg = str(e)[:120]
                print(f"  [WARN] Gemini unavailable (fallback): {safe_msg}")
        if groq_key and not groq_key.startswith("gsk_your"):
            from langchain_groq import ChatGroq
            print("  LLM: Groq (llama-3.3-70b-versatile) を使用")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=groq_key
            )
        elif openai_key and not openai_key.startswith("sk-your"):
            from langchain_openai import OpenAI
            print("  LLM: OpenAI を使用")
            return OpenAI(
                temperature=0,
                openai_api_key=openai_key
            )
        else:
            raise ValueError(
                "LLM APIキーが未設定です。\n"
                ".envファイルに GOOGLE_API_KEY, GROQ_API_KEY, "
                "または OPENAI_API_KEY を設定してください。"
            )

    def load_data(self, data_path: str = None) -> None:
        """
        温泉テキストデータを読み込み、ベクトルDBに保存

        「■」見出しを考慮したセパレータで分割することで、
        意味的なまとまりを保ったチャンクを作成する。

        Args:
            data_path: テキストファイルのパス
                      省略時は data/onsen_knowledge.txt を使用
        """
        file_path = data_path or DEFAULT_DATA_PATH

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"温泉データファイルが見つかりません: {file_path}\n"
                "data/onsen_knowledge.txt を確認してください。"
            )

        # テキストファイルを読み込み
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"[LOAD] Onsen data loaded: {len(text)} chars")

        # Documentオブジェクトに変換
        document = Document(page_content=text)

        # トークンベースのテキスト分割（400〜500 tokens、オーバーラップ10〜20%）
        # separators の優先順位：
        #   "■ "    → 見出し区切り（最も重要な意味の区切り）
        #   "\n\n"  → 段落区切り
        #   "\n"    → 改行
        #   "。"    → 文末（日本語）
        #   "、"    → 読点
        #   " "     → スペース
        #   ""      → 文字単位（最終手段）
        text_splitter = create_token_text_splitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splits = text_splitter.split_documents([document])

        print(f"[SPLIT] {len(splits)} chunks (chunk_size={self.chunk_size} tokens)")

        # チャンク内容をプレビュー表示（デバッグ用）
        for i, split in enumerate(splits):
            preview = split.page_content[:60].replace("\n", " ")
            safe_preview = preview.encode("ascii", errors="replace").decode()
            print(f"  [{i+1}] {safe_preview}...")

        # BM25キーワード検索用にドキュメントを保持
        self.documents = splits

        # ベクトルDBに保存
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

        print(f"\n[OK] Vector DB saved (hybrid search ready)")

    def load_json_chunks(
        self,
        json_path: str | list[str] = None,
    ) -> None:
        """
        JSON形式のチャンクデータを読み込み、Vector DBに格納する。

        メタデータ付きの構造化チャンク（chunk_id, metadata, section, content）を
        ChromaDBに保存。検索時はメタデータでフィルタリング可能。
        chunk_idは全ファイルで一意になるよう連番で統一（chunk_001〜）。

        Args:
            json_path: JSONファイルのパス（単一またはリスト）
                      省略時は data/kusatsu_chunks.json のみ使用
                      複数指定時は DEFAULT_JSON_CHUNK_PATHS で草津・箱根を一括読み込み

        使用例:
            rag = OnsenRAG()
            rag.load_json_chunks()  # 草津のみ
            rag.load_json_chunks(DEFAULT_JSON_CHUNK_PATHS)  # 草津+箱根
        """
        paths = json_path if isinstance(json_path, list) else [json_path or DEFAULT_KUSATSU_CHUNKS_PATH]
        all_chunks = []

        for file_path in paths:
            if not os.path.exists(file_path):
                print(f"[SKIP] Not found: {file_path}")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise FileNotFoundError(
                "読み込めるJSONチャンクファイルがありません。\n"
                "data/kusatsu_chunks.json, hakone_chunks.json, beppu_chunks.json, arima_chunks.json を確認してください。"
            )

        # ChromaDBのメタデータは str/int/float/bool のみ対応のため、
        # tags/keywords, category, area（list）はカンマ区切り文字列に変換
        def _to_str(val):
            if isinstance(val, list):
                return ",".join(str(v) for v in val)
            return str(val) if val is not None else ""

        documents = []
        for chunk in all_chunks:
            meta = chunk.get("metadata", {})
            tags_raw = meta.get("tags") or meta.get("keywords", [])
            tags_str = _to_str(tags_raw) if tags_raw else ""
            # chunk_idプレフィックスを location メタデータとして格納
            chunk_id = chunk.get("chunk_id", "")
            location = chunk_id.split("_")[0] if chunk_id else "unknown"
            doc_metadata = {
                "chunk_id": chunk_id,
                "source": meta.get("source", ""),
                "category": _to_str(meta.get("category", "")),
                "section": chunk.get("section", ""),
                "area": _to_str(meta.get("area", "")),
                "tags": tags_str,
                "location": location,
            }
            doc = Document(
                page_content=chunk.get("content", ""),
                metadata=doc_metadata
            )
            documents.append(doc)

        print(f"[LOAD] JSON chunks loaded: {len(documents)}")

        # BM25キーワード検索用にドキュメントを保持
        self.documents = documents

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        print(f"[OK] Vector DB saved (hybrid search ready)")

    def load_from_data_folder(
        self,
        txt_paths: list[str] = None,
        json_paths: list[str] = None,
    ) -> None:
        """
        RAG/data フォルダ内の全データを読み込み、統合してVector DBに格納

        テキストファイル（onsen_knowledge.txt, beppu.txt 等）と
        JSONチャンク（草津・箱根・別府・有馬）を一括読み込みし、
        統合した知識ベースで検索可能にする。

        Args:
            txt_paths: 読み込むテキストファイルのパスリスト
                      省略時は DEFAULT_TXT_PATHS（onsen_knowledge + beppu）
            json_paths: 読み込むJSONチャンクのパスリスト
                       省略時は DEFAULT_JSON_CHUNK_PATHS（4温泉地）
        """
        txt_paths = txt_paths or DEFAULT_TXT_PATHS
        json_paths = json_paths or DEFAULT_JSON_CHUNK_PATHS

        text_splitter = create_token_text_splitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        all_documents = []

        # テキストファイルを読み込み・分割
        for file_path in txt_paths:
            if not os.path.exists(file_path):
                print(f"[SKIP] Not found: {file_path}")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            doc = Document(page_content=text, metadata={"source": os.path.basename(file_path)})
            splits = text_splitter.split_documents([doc])
            all_documents.extend(splits)
            print(f"[LOAD] {os.path.basename(file_path)}: {len(splits)} chunks")

        # JSONチャンクを読み込み
        def _to_str(val):
            if isinstance(val, list):
                return ",".join(str(v) for v in val)
            return str(val) if val is not None else ""

        for file_path in json_paths:
            if not os.path.exists(file_path):
                print(f"[SKIP] Not found: {file_path}")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                tags_raw = meta.get("tags") or meta.get("keywords", [])
                tags_str = _to_str(tags_raw) if tags_raw else ""
                # chunk_idのプレフィックス（"_"より前）を location として格納
                # 例: "kusatsu_001" → "kusatsu", "arima_010" → "arima"
                # フィルタリング検索で温泉地ごとの絞り込みに使用する
                chunk_id = chunk.get("chunk_id", "")
                location = chunk_id.split("_")[0] if chunk_id else "unknown"
                doc_metadata = {
                    "chunk_id": chunk_id,
                    "source": meta.get("source", os.path.basename(file_path)),
                    "category": _to_str(meta.get("category", "")),
                    "section": chunk.get("section", ""),
                    "area": _to_str(meta.get("area", "")),
                    "tags": tags_str,
                    "location": location,
                }
                doc = Document(
                    page_content=chunk.get("content", ""),
                    metadata=doc_metadata,
                )
                all_documents.append(doc)
            print(f"[LOAD] {os.path.basename(file_path)}: {len(chunks)} chunks")

        if not all_documents:
            raise FileNotFoundError(
                "読み込めるデータがありません。\n"
                f"data フォルダ（{DATA_DIR}）を確認してください。"
            )

        # BM25キーワード検索用にドキュメントを保持
        self.documents = all_documents

        self.vectorstore = Chroma.from_documents(
            documents=all_documents,
            embedding=self.embeddings,
        )
        print(f"[OK] Total {len(all_documents)} chunks loaded into Vector DB (hybrid search ready)")

    def _detect_location(self, question: str) -> str | None:
        """
        質問文から温泉地名を検出し、対応するchunk_idプレフィックスを返す。

        なぜ必要か：
        - 「草津のカフェ」と聞いたのに有馬のチャンクが混ざる問題を防ぐ
        - 検出された温泉地のチャンクのみに絞り込むことで検索精度が向上する

        判定ロジック：
        - 1つの温泉地名だけ検出 → その温泉地でフィルタリング
        - 複数検出 or 検出なし → フィルタリングなし（全チャンクから検索）

        Args:
            question: ユーザーの質問文

        Returns:
            str: 温泉地のプレフィックス（例: "kusatsu"）。検出なしはNone
        """
        detected = []
        for location, keywords in LOCATION_KEYWORDS.items():
            if any(kw in question for kw in keywords):
                detected.append(location)

        # 1つだけ検出された場合のみフィルタリング
        # 複数検出時はどちらも必要な可能性があるためフィルタなし
        if len(detected) == 1:
            return detected[0]
        return None

    def _rerank(
        self,
        question: str,
        documents: list[Document],
        top_k: int = 3,
    ) -> list[Document]:
        """
        CrossEncoderで文書を再ランキングし、上位top_k件を返す。

        なぜ必要か：
        - 初期検索（Bi-Encoder / BM25）は高速だが精度がやや劣る
        - CrossEncoderはクエリと文書のペアを直接比較するため高精度
        - 初期検索で多めに取得 → Re-rankingで絞る戦略が精度向上に有効

        Args:
            question: ユーザーの質問文
            documents: 初期検索で取得した文書リスト
            top_k: 返す文書の最大数

        Returns:
            list[Document]: スコア上位top_k件のドキュメント
        """
        if not documents:
            return []

        # クエリと各文書のペアを作成し、CrossEncoderでスコアリング
        pairs = [[question, doc.page_content] for doc in documents]
        scores = self.cross_encoder.predict(pairs)

        # スコアと文書を紐付け、スコア降順でソート
        doc_score_pairs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # 上位top_k件を採用
        reranked = doc_score_pairs[:top_k]

        print(f"[RERANK] {len(documents)}件 → 上位{len(reranked)}件を採用")
        for i, (doc, score) in enumerate(reranked):
            cid = doc.metadata.get("chunk_id", "?")
            print(f"  [{i+1}] score={score:.4f} chunk_id={cid}")

        return [doc for doc, _ in reranked]

    def _hybrid_search(self, question: str, k: int = 3) -> list[Document]:
        """
        ハイブリッド検索 + Re-ranking パイプライン

        処理の流れ：
          1. セマンティック検索（initial_k件） + BM25キーワード検索（initial_k件）
          2. RRF（Reciprocal Rank Fusion）で両結果を統合
          3. CrossEncoderで再スコアリング（Re-ranking）
          4. 上位final_k件のみ返す

        なぜこのパイプラインか：
        - セマンティック検索: 意味は理解するが、固有名詞（店名・施設名）に弱い
        - BM25キーワード検索: 固有名詞に強いが、類義語・言い換えに弱い
        - RRF: 両方の結果を重み付きで統合し、互いの弱点を補完
        - Re-ranking: 統合結果をCrossEncoderで高精度に再評価し、最終精度を向上

        Args:
            question: ユーザーの質問文
            k: 最終的に返す検索結果の件数（Re-ranking後の採用数）

        Returns:
            list[Document]: Re-ranking後の上位k件のドキュメント
        """
        # 温泉地名を検出してフィルタリング条件を決定
        location = self._detect_location(question)

        # 初期検索は多めに取得（Re-rankingで絞るため）
        search_k = self.initial_k

        # --- セマンティック検索（ベクトル類似度） ---
        semantic_kwargs = {"k": search_k}
        if location:
            semantic_kwargs["filter"] = {
                "$or": [
                    {"location": {"$eq": location}},
                    {"location": {"$eq": "onsen"}},
                ]
            }
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs=semantic_kwargs
        )
        semantic_docs = vector_retriever.invoke(question)

        # --- BM25キーワード検索（単語一致度） ---
        # 温泉地フィルタリングが必要な場合、対象ドキュメントを絞り込む
        bm25_target_docs = self.documents
        if location:
            bm25_target_docs = [
                doc for doc in self.documents
                if doc.metadata.get("location") in (location, "onsen")
            ]

        # BM25検索の実行（対象ドキュメントが空の場合はスキップ）
        bm25_docs = []
        if bm25_target_docs:
            bm25_retriever = BM25Retriever.from_documents(bm25_target_docs)
            bm25_retriever.k = search_k
            bm25_docs = bm25_retriever.invoke(question)

        # --- RRF（Reciprocal Rank Fusion）で結果を統合 ---
        # 各ドキュメントのスコアを page_content をキーにして集計
        RRF_K = 60  # RRF標準定数
        doc_scores = {}  # key: page_content, value: 累積スコア
        doc_map = {}     # key: page_content, value: Document

        # セマンティック検索結果のスコア計算
        for rank, doc in enumerate(semantic_docs):
            content = doc.page_content
            score = self.semantic_weight / (rank + RRF_K)
            doc_scores[content] = doc_scores.get(content, 0) + score
            doc_map[content] = doc

        # BM25検索結果のスコア計算
        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            score = self.keyword_weight / (rank + RRF_K)
            doc_scores[content] = doc_scores.get(content, 0) + score
            doc_map[content] = doc

        # スコア降順でソートし、RRF統合結果を取得
        sorted_contents = sorted(
            doc_scores.keys(),
            key=lambda c: doc_scores[c],
            reverse=True,
        )
        rrf_results = [doc_map[c] for c in sorted_contents]

        loc_label = f"location={location} | " if location else ""
        print(f"[HYBRID] {loc_label}"
              f"semantic={len(semantic_docs)}件 + BM25={len(bm25_docs)}件 "
              f"→ RRF統合={len(rrf_results)}件")

        # --- Re-ranking（CrossEncoderで高精度再評価） ---
        # RRF統合結果をCrossEncoderで再スコアリングし、上位k件のみ返す
        final_results = self._rerank(question, rrf_results, top_k=k)

        return final_results

    # RAG専用プロンプトテンプレート（チャンクID付き・根拠明示形式）
    PROMPT_TEMPLATE = """あなたはRAGシステム専用の日本語質問応答アシスタントです。
以下の【検索結果】に含まれる情報のみを使用して【質問】に回答してください。

【厳守ルール】
- 検索結果に含まれない情報は一切使用しない
- 推測・一般論・補足説明は禁止
- 不明な場合は必ず「該当情報なし」と回答する
- 回答は指定フォーマットを厳守する
- 最大300トークン以内で出力する

【検索結果】（チャンクID付き）
{context}

【質問】
{question}

【回答フォーマット】
回答:
- （簡潔な回答を箇条書きで最大5項目）
  ※ 各項目は2行以内

根拠チャンクID:
- chunk_id_1
- chunk_id_2
"""

    def query(self, question: str, k: int = 3) -> dict:
        """
        温泉に関する質問に対してRAGで回答を生成

        Args:
            question: 質問文（日本語）
                     例: "冬におすすめの温泉地は？"
            k: 検索結果の件数（デフォルト3件）

        Returns:
            dict: 回答結果
                - "result": LLMが生成した回答テキスト
                - "source_documents": 参照したDocumentリスト
                - "chunk_ids": 参照したチャンクIDリスト
        """
        if self.vectorstore is None:
            raise ValueError(
                "データが未読み込みです。先にload_data()を実行してください。"
            )

        # ハイブリッド検索（セマンティック + BM25キーワード + 温泉地フィルタ）
        docs = self._hybrid_search(question, k=k)

        # チャンクID付きでコンテキストを構築
        context_parts = []
        chunk_ids = []
        for doc in docs:
            cid = doc.metadata.get("chunk_id", "")
            if not cid and "source" in doc.metadata:
                # テキスト由来のチャンクは source を ID 代わりに
                cid = doc.metadata.get("source", "unknown").replace(".", "_")
            if not cid:
                cid = f"doc_{len(context_parts) + 1}"
            chunk_ids.append(cid)
            context_parts.append(f"chunk_id: {cid}\n{doc.page_content}")

        context = "\n\n".join(context_parts) if context_parts else "（検索結果なし）"

        prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        chain = prompt | self.llm

        response = chain.invoke({"context": context, "question": question})
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "result": answer.strip(),
            "source_documents": docs,
            "chunk_ids": chunk_ids,
        }

    def search_chunks(self, question: str, k: int = 3) -> list:
        """
        質問に対してどのチャンクが検索されるかを確認（評価用）

        RAGの精度を確認するために使用する。
        LLMの回答生成は行わず、検索結果のみを返す。

        Args:
            question: 検索クエリ
            k: 取得件数

        Returns:
            list: 検索結果のDocumentリスト
        """
        if self.vectorstore is None:
            raise ValueError("データが未読み込みです。")

        # ハイブリッド検索（セマンティック + BM25キーワード + 温泉地フィルタ）
        results = self._hybrid_search(question, k=k)

        print(f"\n[SEARCH] Question: 「{question}」")
        print(f"   検索結果: {len(results)}件")
        for i, doc in enumerate(results):
            content = doc.page_content.replace("\n", " ")
            preview = content[:100] + "..." \
                if len(content) > 100 else content
            safe_preview = preview.encode("ascii", errors="replace").decode()
            print(f"  [{i+1}] {safe_preview}")

        return results

    def evaluate(self, questions_path: str = None) -> list:
        """
        サンプル質問を使ってRAGの精度を一括評価

        data/sample_questions.json の質問を順番に実行し、
        検索されたチャンクと期待されるチャンクを比較する。

        Args:
            questions_path: 質問ファイルのパス（省略時はデフォルト）

        Returns:
            list: 評価結果のリスト
        """
        if self.vectorstore is None:
            raise ValueError("データが未読み込みです。")

        file_path = questions_path or DEFAULT_QUESTIONS_PATH

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

            # 検索実行
            docs = self.search_chunks(question, k=3)

            # キーワードが検索結果に含まれているか確認
            all_text = " ".join([d.page_content for d in docs])
            matched_keywords = [
                kw for kw in keywords if kw in all_text
            ]
            # マッチ率を計算（0.0〜1.0）
            match_rate = len(matched_keywords) / len(keywords) \
                if keywords else 0.0

            # 結果を判定（cp932対応で絵文字を避ける）
            status = "[OK]" if match_rate >= 0.5 else "[WARN]" \
                if match_rate > 0 else "[BAD]"

            print(f"   {status} キーワード一致率: {match_rate:.0%} "
                  f"({len(matched_keywords)}/{len(keywords)})")
            print(f"   期待チャンク: {expected}")
            print()

            results.append({
                "question": question,
                "expected_chunk": expected,
                "match_rate": match_rate,
                "matched_keywords": matched_keywords,
                "status": status
            })

        # サマリー表示
        total = len(results)
        good = sum(1 for r in results if r["status"] == "[OK]")
        warn = sum(1 for r in results if r["status"] == "[WARN]")
        bad = sum(1 for r in results if r["status"] == "[BAD]")

        print("=" * 60)
        print(f"Summary: OK={good} / WARN={warn} / BAD={bad} "
              f"（全{total}問）")
        print("=" * 60)

        return results
