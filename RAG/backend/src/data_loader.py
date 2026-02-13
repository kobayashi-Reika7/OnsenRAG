"""
OnsenRAG データローダー
========================
JSON チャンクファイルの読み込みと Document オブジェクトへの変換を担当する。
"""

import os
import json

from langchain_core.documents import Document


def meta_to_str(val) -> str:
    """
    ChromaDB 互換のメタデータ値変換。

    ChromaDB のメタデータは str/int/float/bool のみ対応のため、
    list 型はカンマ区切り文字列に変換する。
    """
    if isinstance(val, list):
        return ",".join(str(v) for v in val)
    return str(val) if val is not None else ""


def load_json_chunks(paths: list[str]) -> list[Document]:
    """
    JSON チャンクファイルを読み込み、Document リストを返す。

    各チャンクの chunk_id プレフィックスを location メタデータとして付与し、
    温泉地別フィルタリングを可能にする。

    Args:
        paths: JSON ファイルパスのリスト

    Returns:
        list[Document]: メタデータ付き Document のリスト

    Raises:
        FileNotFoundError: 読み込めるファイルがない場合
    """
    all_chunks: list[dict] = []

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
            "data/ フォルダ内の JSON ファイルを確認してください。"
        )

    documents: list[Document] = []
    for chunk in all_chunks:
        meta = chunk.get("metadata", {})
        tags_raw = meta.get("tags") or meta.get("keywords", [])
        tags_str = meta_to_str(tags_raw) if tags_raw else ""

        chunk_id = chunk.get("chunk_id", "")
        location = chunk_id.split("_")[0] if chunk_id else "unknown"

        doc = Document(
            page_content=chunk.get("content", ""),
            metadata={
                "chunk_id": chunk_id,
                "source": meta.get("source", ""),
                "category": meta_to_str(meta.get("category", "")),
                "section": chunk.get("section", ""),
                "area": meta_to_str(meta.get("area", "")),
                "tags": tags_str,
                "location": location,
            },
        )
        documents.append(doc)

    print(f"[LOAD] JSON chunks loaded: {len(documents)}")
    return documents


def load_txt_files(paths: list[str], text_splitter) -> list[Document]:
    """
    テキストファイルを読み込み、分割して Document リストを返す。

    Args:
        paths: テキストファイルパスのリスト
        text_splitter: LangChain のテキストスプリッター

    Returns:
        list[Document]: 分割済み Document のリスト
    """
    documents: list[Document] = []

    for file_path in paths:
        if not os.path.exists(file_path):
            print(f"[SKIP] Not found: {file_path}")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc = Document(
            page_content=text,
            metadata={"source": os.path.basename(file_path)},
        )
        splits = text_splitter.split_documents([doc])
        documents.extend(splits)
        print(f"[LOAD] {os.path.basename(file_path)}: {len(splits)} chunks")

    return documents
