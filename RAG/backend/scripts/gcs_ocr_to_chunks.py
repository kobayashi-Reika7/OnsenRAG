"""
GCS OCR to RAG Pipeline
=======================
GCS output/ (または PDF 別サブフォルダ) の Vision API JSON から
fullTextAnnotation.text を取得し、ソース PDF ごとにグループ化して
正規化・チャンク分割・JSON 保存する。

実行:
  python gcs_ocr_to_chunks.py                    # output/ 全体を処理
  python gcs_ocr_to_chunks.py output/arima/       # 特定プレフィクスのみ

前提:
  GOOGLE_APPLICATION_CREDENTIALS が設定済み
"""
import os
import sys
import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

from google.cloud import storage
from langchain_core.documents import Document

from src.text_splitter_utils import create_token_text_splitter

# GCS
BUCKET_NAME = "myocr_1"
DEFAULT_PREFIX = "output/"
DATA_DIR = Path(__file__).resolve().parent / "data"

# ソース PDF 名 → チャンク ID / 出力ファイル名のマッピング
SOURCE_MAP = {
    "有馬マップ.pdf":       {"id_prefix": "arima_ocr",   "area": "有馬温泉",  "file": "arima_ocr_chunks.json"},
    "別府地獄.pdf":         {"id_prefix": "beppu_j_ocr", "area": "別府温泉",  "file": "beppu_jigoku_ocr_chunks.json"},
    "別府温泉.pdf":         {"id_prefix": "beppu_ocr",   "area": "別府温泉",  "file": "beppu_ocr_chunks.json"},
    "箱根温泉.pdf":         {"id_prefix": "hakone_ocr",  "area": "箱根温泉",  "file": "hakone_ocr_chunks.json"},
    "草津温泉ガイド.pdf":   {"id_prefix": "kusatsu_g_ocr", "area": "草津温泉", "file": "kusatsu_guide_ocr_chunks.json"},
    "草津マップ.pdf":       {"id_prefix": "kusatsu_m_ocr", "area": "草津温泉", "file": "kusatsu_map_ocr_chunks.json"},
}


def fetch_and_group_jsons(
    bucket_name: str = BUCKET_NAME,
    prefix: str = DEFAULT_PREFIX,
) -> dict[str, list[dict]]:
    """
    GCS の JSON をダウンロードし、ソース PDF ごとにグループ化して返す。
    Returns: {pdf_filename: [json_data, ...]}
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    grouped: dict[str, list[dict]] = defaultdict(list)

    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".json"):
            continue
        content = blob.download_as_string().decode("utf-8")
        data = json.loads(content)
        uri = (data.get("inputConfig", {})
                   .get("gcsSource", {})
                   .get("uri", ""))
        pdf_name = uri.split("/")[-1] if uri else "unknown"
        grouped[pdf_name].append(data)

    return dict(grouped)


def extract_full_text(jsons: list[dict]) -> str:
    """各 JSON から fullTextAnnotation.text を取得して結合する。"""
    texts = []
    for data in jsons:
        for resp in data.get("responses", []):
            ft = resp.get("fullTextAnnotation")
            if ft and ft.get("text"):
                texts.append(ft["text"])
    return "\n\n".join(texts)


def normalize_text(text: str) -> str:
    """NFKC 正規化 + 連続空白・改行を整理。"""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, document_type: str = "general") -> list[Document]:
    """テキストをチャンク分割して Document のリストにする。"""
    splitter = create_token_text_splitter(document_type=document_type)
    doc = Document(page_content=text)
    return splitter.split_documents([doc])


def chunks_to_json(
    chunks: list[Document],
    id_prefix: str,
    source: str,
    area: str,
) -> list[dict]:
    """Document を既存チャンク JSON と同形式に変換。"""
    out = []
    for i, doc in enumerate(chunks, 1):
        out.append({
            "chunk_id": f"{id_prefix}_{i:03d}",
            "section": "",
            "metadata": {
                "source": source,
                "category": ["OCR"],
                "area": [area],
                "keywords": [],
            },
            "content": doc.page_content,
        })
    return out


def process_one_pdf(pdf_name: str, jsons: list[dict]) -> str | None:
    """
    1つの PDF の OCR JSON を処理してチャンク JSON を保存する。
    Returns: 保存先パス or None
    """
    cfg = SOURCE_MAP.get(pdf_name)
    if not cfg:
        safe = pdf_name.encode("ascii", errors="replace").decode()
        print(f"  [SKIP] Unknown source: {safe}")
        return None

    raw = extract_full_text(jsons)
    if not raw:
        print(f"  [SKIP] No text: {pdf_name}")
        return None

    text = normalize_text(raw)
    chunks = chunk_text(text)

    json_chunks = chunks_to_json(
        chunks,
        id_prefix=cfg["id_prefix"],
        source=pdf_name,
        area=cfg["area"],
    )

    out_path = DATA_DIR / cfg["file"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_chunks, f, ensure_ascii=False, indent=2)

    safe_name = pdf_name.encode("ascii", errors="replace").decode()
    print(f"  {safe_name}: {len(raw)} chars -> {len(text)} normalized -> {len(chunks)} chunks -> {out_path.name}")
    return str(out_path)


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREFIX

    print("[1/3] Fetching OCR JSONs from gs://%s/%s ..." % (BUCKET_NAME, prefix))
    grouped = fetch_and_group_jsons(prefix=prefix)
    if not grouped:
        print("  No JSON found.")
        return
    print("  Found %d source PDF(s), %d file(s) total" % (
        len(grouped),
        sum(len(v) for v in grouped.values()),
    ))

    print("[2/3] Processing per source PDF...")
    saved_files = []
    for pdf_name, jsons in sorted(grouped.items()):
        result = process_one_pdf(pdf_name, jsons)
        if result:
            saved_files.append(result)

    if not saved_files:
        print("  No chunks generated.")
        return

    print("[3/3] Done!")
    print("  Saved %d file(s):" % len(saved_files))
    for f in saved_files:
        print("    - %s" % f)


if __name__ == "__main__":
    main()
