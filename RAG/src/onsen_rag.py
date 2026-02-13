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
        top_k: int = None,
    ) -> list[tuple[Document, float]]:
        """
        ステップ1: 類似度検索結果をCrossEncoderでスコアリング

        なぜ必要か：
        - 初期検索（Bi-Encoder / BM25）は高速だが精度がやや劣る
        - CrossEncoderはクエリと文書のペアを直接比較するため高精度
        - スコアを保持して後段のLLM候補抽出・最終選択で活用する

        Args:
            question: ユーザーの質問文
            documents: 初期検索で取得した文書リスト
            top_k: 返す文書の最大数（Noneで全件返す）

        Returns:
            list[tuple[Document, float]]: (文書, CrossEncoderスコア)のリスト（スコア降順）
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

        # top_k指定があれば上位のみ返す
        if top_k is not None:
            doc_score_pairs = list(doc_score_pairs)[:top_k]

        result = list(doc_score_pairs)
        print(f"[RERANK] {len(documents)}件をCrossEncoderでスコアリング → {len(result)}件")
        for i, (doc, score) in enumerate(result[:5]):
            cid = doc.metadata.get("chunk_id", "?")
            print(f"  [{i+1}] CE_score={score:.4f} chunk_id={cid}")

        return result

    # LLM候補抽出用プロンプト
    # 各文書の関連度を0〜10で評価し、上位を選定する
    LLM_EXTRACT_PROMPT = """あなたは検索結果の関連度を評価する専門家です。
以下の【質問】に対して、各【候補文書】がどれだけ関連しているかを0〜10の整数で評価してください。

【評価基準】
- 10: 質問に対する直接的な回答が含まれている
- 7-9: 質問に強く関連する情報が含まれている
- 4-6: 部分的に関連する情報がある
- 1-3: ほぼ関連しない
- 0: 完全に無関連

【質問】
{question}

【候補文書】
{candidates}

【回答フォーマット】（必ずこの形式で回答）
chunk_id:スコア
例:
kusatsu_001:8
kusatsu_015:3
"""

    def _llm_extract_candidates(
        self,
        question: str,
        scored_docs: list[tuple[Document, float]],
        top_k: int = 5,
    ) -> list[tuple[Document, float, float]]:
        """
        ステップ2: LLMで各候補文書の関連度を評価し、上位top_k件を抽出

        なぜLLM評価が有効か：
        - CrossEncoderは汎用的な文書関連度を測定するが、質問の意図を深く理解しない
        - LLMは質問の意図を理解し、「本当に回答に使える情報か」を判断できる
        - 例: 「草津のカフェ」→ CrossEncoderは"草津"を含む全文書を高スコアにするが、
          LLMは「カフェ情報」を含む文書のみを高く評価する

        Args:
            question: ユーザーの質問文
            scored_docs: (文書, CrossEncoderスコア)のリスト
            top_k: LLM評価後に返す上位件数

        Returns:
            list[tuple[Document, float, float]]:
                (文書, CrossEncoderスコア, LLMスコア)のリスト（LLMスコア降順）
        """
        if not scored_docs:
            return []

        # 候補文書をフォーマット
        candidates_text = ""
        doc_map = {}  # chunk_id → (Document, CE_score) のマップ
        for i, (doc, ce_score) in enumerate(scored_docs):
            cid = doc.metadata.get("chunk_id", f"doc_{i+1}")
            # LLMに送るコンテキスト（長すぎる場合は切り詰め）
            content = doc.page_content[:300]
            candidates_text += f"\n[{cid}]\n{content}\n"
            doc_map[cid] = (doc, ce_score)

        # LLMに候補評価を依頼
        prompt = PromptTemplate(
            template=self.LLM_EXTRACT_PROMPT,
            input_variables=["question", "candidates"]
        )
        chain = prompt | self.llm

        try:
            response = chain.invoke({
                "question": question,
                "candidates": candidates_text,
            })
            response_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            # LLM呼び出し失敗時はCrossEncoderスコアのみで上位を返す
            print(f"[LLM_EXTRACT] LLM評価失敗、CEスコアで代替: {str(e)[:80]}")
            return [
                (doc, ce_score, 0.0)
                for doc, ce_score in scored_docs[:top_k]
            ]

        # LLMのレスポンスからスコアを解析
        # フォーマット: "chunk_id:スコア" の各行をパース
        llm_scores = {}
        for line in response_text.strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.rsplit(":", 1)
            if len(parts) != 2:
                continue
            cid_part = parts[0].strip()
            try:
                score_val = float(parts[1].strip())
                # 0〜10の範囲にクリップ
                score_val = max(0.0, min(10.0, score_val))
                llm_scores[cid_part] = score_val
            except ValueError:
                continue

        # (Document, CEスコア, LLMスコア)を構築
        results = []
        for cid, (doc, ce_score) in doc_map.items():
            llm_score = llm_scores.get(cid, 0.0)
            results.append((doc, ce_score, llm_score))

        # LLMスコア降順でソートし、上位top_k件を返す
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:top_k]

        print(f"[LLM_EXTRACT] {len(scored_docs)}件 → LLM評価で上位{len(results)}件を抽出")
        for i, (doc, ce_score, llm_score) in enumerate(results):
            cid = doc.metadata.get("chunk_id", "?")
            print(f"  [{i+1}] LLM={llm_score:.0f}/10 CE={ce_score:.4f} chunk_id={cid}")

        return results

    def _final_selection(
        self,
        candidates: list[tuple[Document, float, float]],
        top_k: int = 3,
        ce_weight: float = 0.4,
        llm_weight: float = 0.6,
    ) -> list[Document]:
        """
        ステップ3: CrossEncoderスコアとLLMスコアを統合し、最終的な上位文書を選定

        スコア統合の計算式：
          final_score = ce_weight * normalize(CE_score) + llm_weight * normalize(LLM_score)

        なぜ統合するのか：
        - CrossEncoder: 文書レベルの汎用的な関連度（固有名詞や文体の一致に強い）
        - LLM: 質問の意図を理解した上での関連度（回答に使える情報かを判断）
        - 両方のスコアを統合することで、最も適切な文書を選定できる

        Args:
            candidates: (文書, CrossEncoderスコア, LLMスコア)のリスト
            top_k: 最終的に返す件数
            ce_weight: CrossEncoderスコアの重み（デフォルト0.4）
            llm_weight: LLMスコアの重み（デフォルト0.6）

        Returns:
            list[Document]: 最終選定された上位top_k件のドキュメント
        """
        if not candidates:
            return []

        # CrossEncoderスコアを0〜1に正規化
        ce_scores = [ce for _, ce, _ in candidates]
        ce_min, ce_max = min(ce_scores), max(ce_scores)
        ce_range = ce_max - ce_min if ce_max != ce_min else 1.0

        # LLMスコアを0〜1に正規化（元は0〜10）
        # 統合スコアを計算
        final_scored = []
        for doc, ce_score, llm_score in candidates:
            ce_norm = (ce_score - ce_min) / ce_range
            llm_norm = llm_score / 10.0
            final_score = ce_weight * ce_norm + llm_weight * llm_norm
            final_scored.append((doc, final_score, ce_score, llm_score))

        # 統合スコア降順でソート
        final_scored.sort(key=lambda x: x[1], reverse=True)
        top_results = final_scored[:top_k]

        print(f"[FINAL] スコア統合（CE×{ce_weight} + LLM×{llm_weight}）→ 上位{len(top_results)}件を最終選定")
        for i, (doc, final, ce, llm) in enumerate(top_results):
            cid = doc.metadata.get("chunk_id", "?")
            print(f"  [{i+1}] final={final:.4f} (CE={ce:.4f} LLM={llm:.0f}/10) chunk_id={cid}")

        return [doc for doc, _, _, _ in top_results]

    def _hybrid_search(self, question: str, k: int = 3) -> list[Document]:
        """
        3段階検索パイプライン：類似度検索・スコアリング → LLM候補抽出 → 最終選択

        処理の流れ：
          0. 温泉地フィルタ（質問から地名検出 → 該当チャンクのみ対象）
          1. 類似度検索・スコアリング
             - セマンティック検索（initial_k件）+ BM25キーワード検索（initial_k件）
             - RRF（Reciprocal Rank Fusion）で統合
             - CrossEncoderで全候補をスコアリング
          2. LLM候補抽出（上位5件）
             - LLMが各候補の関連度を0〜10で評価
             - 質問の意図を理解した上で本当に有用な文書を選定
          3. 最終選択（スコア統合）
             - CrossEncoderスコア × 0.4 + LLMスコア × 0.6 で統合
             - 上位k件を最終回答用に選定

        Args:
            question: ユーザーの質問文
            k: 最終的に返す検索結果の件数

        Returns:
            list[Document]: 3段階選定を経た上位k件のドキュメント
        """
        # 温泉地名を検出してフィルタリング条件を決定
        location = self._detect_location(question)

        # 初期検索は多めに取得（後段で絞るため）
        search_k = self.initial_k

        # ========================================
        # ステップ1: 類似度検索、スコアリング
        # ========================================

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
        bm25_target_docs = self.documents
        if location:
            bm25_target_docs = [
                doc for doc in self.documents
                if doc.metadata.get("location") in (location, "onsen")
            ]

        bm25_docs = []
        if bm25_target_docs:
            bm25_retriever = BM25Retriever.from_documents(bm25_target_docs)
            bm25_retriever.k = search_k
            bm25_docs = bm25_retriever.invoke(question)

        # --- RRF（Reciprocal Rank Fusion）で統合 ---
        RRF_K = 60
        doc_scores = {}
        doc_map = {}

        for rank, doc in enumerate(semantic_docs):
            content = doc.page_content
            score = self.semantic_weight / (rank + RRF_K)
            doc_scores[content] = doc_scores.get(content, 0) + score
            doc_map[content] = doc

        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            score = self.keyword_weight / (rank + RRF_K)
            doc_scores[content] = doc_scores.get(content, 0) + score
            doc_map[content] = doc

        sorted_contents = sorted(
            doc_scores.keys(),
            key=lambda c: doc_scores[c],
            reverse=True,
        )
        rrf_results = [doc_map[c] for c in sorted_contents]

        loc_label = f"location={location} | " if location else ""
        print(f"[STEP1] 類似度検索 {loc_label}"
              f"semantic={len(semantic_docs)}件 + BM25={len(bm25_docs)}件 "
              f"→ RRF統合={len(rrf_results)}件")

        # --- CrossEncoderでスコアリング ---
        scored_docs = self._rerank(question, rrf_results)

        # ========================================
        # ステップ2: LLM候補抽出（上位5件）
        # ========================================
        llm_candidates = self._llm_extract_candidates(
            question, scored_docs, top_k=self.final_k + 2
        )

        # ========================================
        # ステップ3: 最終選択（スコア統合）
        # ========================================
        final_results = self._final_selection(llm_candidates, top_k=k)

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
