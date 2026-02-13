"""
OnsenRAG 検索パイプライン（多言語対応）
==========================================
3段階検索パイプライン（ハイブリッド検索 → Re-ranking → LLM評価）のロジックを担当する。

パイプライン概要:
  1. セマンティック検索 + BM25 → RRF統合 → CrossEncoderスコアリング
  2. LLM候補抽出（質問意図を理解した関連度評価）
  3. スコア統合（CE × 0.4 + LLM × 0.6）→ 最終選定
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.config import LOCATION_KEYWORDS, CONFIDENCE_THRESHOLD
from src.prompts import LLM_EXTRACT_PROMPT


# ============================================================
# 温泉地検出
# ============================================================


def detect_location(
    question: str,
    last_location: str | None = None,
    location_keywords: dict[str, list[str]] | None = None,
) -> str | None:
    """
    質問文から温泉地名を検出し、対応する chunk_id プレフィックスを返す。

    判定ロジック:
      - 1つの温泉地名だけ検出 → その温泉地でフィルタリング
      - 複数検出 or 検出なし → フィルタリングなし
      - 検出なし & last_location あり → 会話コンテキスト継続
    """
    kw_map = location_keywords or LOCATION_KEYWORDS

    detected = [
        loc for loc, keywords in kw_map.items()
        if any(kw in question for kw in keywords)
    ]

    if len(detected) == 1:
        return detected[0]

    # 会話コンテキスト継続
    if not detected and last_location:
        print(f"[CONTEXT] No location detected -> continuing previous context: {last_location}")
        return last_location

    return None


# ============================================================
# Re-ranking（CrossEncoder）
# ============================================================


def rerank(
    question: str,
    documents: list[Document],
    cross_encoder,
    top_k: int | None = None,
) -> list[tuple[Document, float]]:
    """
    CrossEncoder で文書をスコアリングし、スコア降順で返す。
    """
    if not documents:
        return []

    pairs = [[question, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)

    doc_score_pairs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    result = list(doc_score_pairs)
    if top_k is not None:
        result = result[:top_k]

    print(f"[RERANK] CrossEncoder scored {len(documents)} docs -> {len(result)} selected")
    for i, (doc, score) in enumerate(result[:5]):
        cid = doc.metadata.get("chunk_id", "?")
        print(f"  [{i+1}] CE_score={score:.4f} chunk_id={cid}")

    return result


# ============================================================
# LLM候補抽出
# ============================================================


def llm_extract_candidates(
    question: str,
    scored_docs: list[tuple[Document, float]],
    llm,
    top_k: int = 5,
) -> list[tuple[Document, float, float]]:
    """
    LLM で各候補文書の関連度を 0〜10 で評価し、上位 top_k 件を返す。
    """
    if not scored_docs:
        return []

    # 候補文書をフォーマット
    candidates_text = ""
    doc_map: dict[str, tuple[Document, float]] = {}
    for i, (doc, ce_score) in enumerate(scored_docs):
        cid = doc.metadata.get("chunk_id", f"doc_{i+1}")
        content = doc.page_content[:300]
        candidates_text += f"\n[{cid}]\n{content}\n"
        doc_map[cid] = (doc, ce_score)

    # LLM に候補評価を依頼
    prompt = PromptTemplate(
        template=LLM_EXTRACT_PROMPT,
        input_variables=["question", "candidates"],
    )
    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": question,
            "candidates": candidates_text,
        })
        response_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"[LLM_EXTRACT] LLM evaluation failed, using CE scores: {str(e)[:80]}")
        return [(doc, ce_score, 0.0) for doc, ce_score in scored_docs[:top_k]]

    # レスポンスからスコアをパース（"chunk_id:スコア" 形式）
    llm_scores: dict[str, float] = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            continue
        cid_part = parts[0].strip()
        try:
            score_val = max(0.0, min(10.0, float(parts[1].strip())))
            llm_scores[cid_part] = score_val
        except ValueError:
            continue

    # (Document, CE_score, LLM_score) を構築
    results = [
        (doc, ce_score, llm_scores.get(cid, 0.0))
        for cid, (doc, ce_score) in doc_map.items()
    ]
    results.sort(key=lambda x: x[2], reverse=True)
    results = results[:top_k]

    print(f"[LLM_EXTRACT] {len(scored_docs)} docs -> top {len(results)} by LLM evaluation")
    for i, (doc, ce_score, llm_score) in enumerate(results):
        cid = doc.metadata.get("chunk_id", "?")
        print(f"  [{i+1}] LLM={llm_score:.0f}/10 CE={ce_score:.4f} chunk_id={cid}")

    return results


# ============================================================
# 最終選択（スコア統合）
# ============================================================


def final_selection(
    candidates: list[tuple[Document, float, float]],
    top_k: int = 3,
    ce_weight: float = 0.4,
    llm_weight: float = 0.6,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[Document]:
    """
    CrossEncoder スコアと LLM スコアを統合し、最終的な上位文書を選定する。
    """
    if not candidates:
        return []

    # 信頼度フィルタ
    confident = [
        (doc, ce, llm) for doc, ce, llm in candidates
        if ce >= confidence_threshold
    ]
    if not confident:
        print(f"[FINAL] All CE scores below threshold ({confidence_threshold}) -> no results")
        return []

    if len(confident) < len(candidates):
        print(f"[FINAL] Confidence filter: {len(candidates)} -> {len(confident)} "
              f"(threshold={confidence_threshold})")

    # CE スコアを 0〜1 に正規化
    ce_scores = [ce for _, ce, _ in confident]
    ce_min, ce_max = min(ce_scores), max(ce_scores)
    ce_range = ce_max - ce_min if ce_max != ce_min else 1.0

    # 統合スコアを計算
    final_scored = []
    for doc, ce_score, llm_score in confident:
        ce_norm = (ce_score - ce_min) / ce_range
        llm_norm = llm_score / 10.0
        score = ce_weight * ce_norm + llm_weight * llm_norm
        final_scored.append((doc, score, ce_score, llm_score))

    final_scored.sort(key=lambda x: x[1], reverse=True)
    top_results = final_scored[:top_k]

    print(f"[FINAL] Score fusion (CE*{ce_weight} + LLM*{llm_weight}) -> top {len(top_results)} selected")
    for i, (doc, final, ce, llm) in enumerate(top_results):
        cid = doc.metadata.get("chunk_id", "?")
        print(f"  [{i+1}] final={final:.4f} (CE={ce:.4f} LLM={llm:.0f}/10) chunk_id={cid}")

    return [doc for doc, _, _, _ in top_results]


# ============================================================
# ハイブリッド検索（統合オーケストレーション）
# ============================================================


def hybrid_search(
    question: str,
    *,
    vectorstore,
    cross_encoder,
    llm,
    semantic_weight: float,
    keyword_weight: float,
    initial_k: int,
    final_k: int,
    bm25_all=None,
    bm25_by_location: dict | None = None,
    last_location: str | None = None,
    k: int = 3,
    location_keywords: dict[str, list[str]] | None = None,
) -> tuple[list[Document], str | None]:
    """
    3段階検索パイプラインを実行する。

    Returns:
        (検索結果の Document リスト, 検出された温泉地名) のタプル
    """
    bm25_by_location = bm25_by_location or {}

    # 温泉地検出
    location = detect_location(question, last_location, location_keywords)

    search_k = initial_k

    # ========================================
    # ステップ1: 類似度検索・スコアリング
    # ========================================

    # セマンティック検索
    semantic_kwargs: dict = {"k": search_k}
    if location:
        semantic_kwargs["filter"] = {
            "$or": [
                {"location": {"$eq": location}},
                {"location": {"$eq": "onsen"}},
            ]
        }
    vector_retriever = vectorstore.as_retriever(search_kwargs=semantic_kwargs)
    semantic_docs = vector_retriever.invoke(question)

    # BM25 キーワード検索
    bm25_docs: list[Document] = []
    if location:
        loc_retriever = bm25_by_location.get(location)
        if loc_retriever:
            bm25_docs = loc_retriever.invoke(question)
        onsen_retriever = bm25_by_location.get("onsen")
        if onsen_retriever:
            bm25_docs.extend(onsen_retriever.invoke(question))
    elif bm25_all:
        bm25_docs = bm25_all.invoke(question)

    # RRF（Reciprocal Rank Fusion）で統合
    RRF_K = 60
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(semantic_docs):
        content = doc.page_content
        doc_scores[content] = doc_scores.get(content, 0) + semantic_weight / (rank + RRF_K)
        doc_map[content] = doc

    for rank, doc in enumerate(bm25_docs):
        content = doc.page_content
        doc_scores[content] = doc_scores.get(content, 0) + keyword_weight / (rank + RRF_K)
        doc_map[content] = doc

    sorted_contents = sorted(doc_scores.keys(), key=lambda c: doc_scores[c], reverse=True)
    rrf_results = [doc_map[c] for c in sorted_contents]

    loc_label = f"location={location} | " if location else ""
    print(f"[STEP1] Similarity search {loc_label}"
          f"semantic={len(semantic_docs)} + BM25={len(bm25_docs)} "
          f"-> RRF merged={len(rrf_results)}")

    # CrossEncoder でスコアリング
    scored_docs = rerank(question, rrf_results, cross_encoder)

    # ========================================
    # ステップ2: CrossEncoderスコアで最終選定
    # ========================================
    llm_candidates = [(doc, ce_score, 0.0) for doc, ce_score in scored_docs]
    print(f"[STEP2] Selecting top {k} by CrossEncoder score (LLM call skipped)")

    # ========================================
    # ステップ3: 最終選択
    # ========================================
    final_results = final_selection(llm_candidates, top_k=k)

    return final_results, location
