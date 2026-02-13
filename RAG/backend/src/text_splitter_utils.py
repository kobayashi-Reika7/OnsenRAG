"""
トークンベーステキスト分割ユーティリティ
==========================================

chunk_size・chunk_overlap を tokens で管理する TextSplitter を提供する。
LLMのコンテキスト制限（トークン数）に合わせた精度の高いチャンク分割が可能。

ドキュメントタイプ別プリセット（精度向上のヒントに基づく）:
+------------------+-----------+-----------+------------------+
| タイプ           | chunk_size| overlap   | 説明              |
+------------------+-----------+-----------+------------------+
| general          | 600       | 75        | 一般的な文書・バランス重視 |
| technical        | 900       | 120       | 技術文書・文脈保持重視   |
| faq              | 350       | 40        | 短い情報(FAQ)・簡潔さ重視 |
| long             | 1000      | 150       | 長文(論文等)・詳細保持重視 |
+------------------+-----------+-----------+------------------+

チャンキングのヒント:
- 意味的な分割を優先（段落・文末・句読点で分割）
- 適切な overlap で文脈の継続性を維持
- メタデータ（ページ番号・ソース）を活用
"""

# langchain-text-splitters または langchain からインポート
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# 日本語トークン化用エンコーディング（GPT-4/Claude互換）
# cl100k_base: 日本語を含む多言語に最適
DEFAULT_ENCODING = "cl100k_base"

# デフォルトのチャンク設定（tokens）- general プリセットと同等
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 75

# ドキュメントタイプ別プリセット（tokens）
# 一般的な文書: 500-1000 chars → バランス重視
# 技術文書: 800-1200 chars → 文脈保持重視
# FAQ: 300-500 chars → 簡潔さ重視
# 長文: 1000-1500 chars → 詳細保持重視
DOCUMENT_TYPE_PRESETS: dict[str, dict[str, int]] = {
    "general": {"chunk_size": 600, "chunk_overlap": 75},   # 一般的な文書
    "technical": {"chunk_size": 900, "chunk_overlap": 120},  # 技術文書
    "faq": {"chunk_size": 350, "chunk_overlap": 40},        # 短い情報(FAQ)
    "long": {"chunk_size": 1000, "chunk_overlap": 150},      # 長文(論文等)
}


def get_chunk_preset(document_type: str = "general") -> tuple[int, int]:
    """
    ドキュメントタイプから chunk_size, chunk_overlap を取得する。

    Args:
        document_type: "general" | "technical" | "faq" | "long"

    Returns:
        (chunk_size, chunk_overlap) のタプル
    """
    preset = DOCUMENT_TYPE_PRESETS.get(
        document_type, DOCUMENT_TYPE_PRESETS["general"]
    )
    return preset["chunk_size"], preset["chunk_overlap"]


def create_token_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    document_type: str | None = None,
    encoding_name: str = DEFAULT_ENCODING,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    トークンベースの RecursiveCharacterTextSplitter を生成する。

    tiktoken を用いてチャンクサイズをトークン数で管理し、
    意味単位（separators）を優先して分割する。

    Args:
        chunk_size: 1チャンクあたりの最大トークン数（None の場合は preset を使用）
        chunk_overlap: チャンク間の重複トークン数（None の場合は preset を使用）
        document_type: プリセット指定 "general"|"technical"|"faq"|"long"
        encoding_name: tiktoken エンコーディング名
        separators: 分割の優先順位。None の場合は日本語向けデフォルトを使用

    Returns:
        RecursiveCharacterTextSplitter: トークン単位で分割するスプリッター

    Example:
        splitter = create_token_text_splitter(document_type="general")
        splitter = create_token_text_splitter(chunk_size=600, chunk_overlap=75)
    """
    if document_type is not None:
        chunk_size, chunk_overlap = get_chunk_preset(document_type)
    else:
        chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP

    if separators is None:
        separators = ["■ ", "\n\n", "\n", "。", "、", " ", ""]

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
