"""
Google Cloud Vision API - PDF OCR
=================================
GCS上のPDFを非同期でOCRし、出力をGCSに保存する。
PDF ごとに出力先サブフォルダを分けて上書きを防ぐ。

前提:
  - GOOGLE_APPLICATION_CREDENTIALS が設定済み
  - GCS bucket (myocr_1) にPDFが存在

実行:
  python run_vision_ocr.py                          # 全PDF一括
  python run_vision_ocr.py 有馬マップ.pdf           # 指定PDFのみ
"""
import sys
from google.cloud import vision

# OCR 対象 PDF リスト
ALL_PDFS = [
    "有馬マップ.pdf",
    "別府地獄.pdf",
    "別府温泉.pdf",
    "箱根温泉.pdf",
    "草津温泉ガイド.pdf",
    "草津マップ.pdf",
]

BUCKET = "myocr_1"


def run_ocr(pdf_name: str):
    """1つのPDFに対してOCRを実行する。出力先は output/{stem}/ に分離。"""
    client = vision.ImageAnnotatorClient()

    stem = pdf_name.replace(".pdf", "")
    gcs_source = f"gs://{BUCKET}/{pdf_name}"
    gcs_dest = f"gs://{BUCKET}/output/{stem}/"

    feature = vision.Feature(
        type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
    )
    input_config = vision.InputConfig(
        gcs_source=vision.GcsSource(uri=gcs_source),
        mime_type="application/pdf",
    )
    output_config = vision.OutputConfig(
        gcs_destination=vision.GcsDestination(uri=gcs_dest),
        batch_size=1,
    )
    request = vision.AsyncAnnotateFileRequest(
        features=[feature],
        input_config=input_config,
        output_config=output_config,
    )

    safe = pdf_name.encode("ascii", errors="replace").decode()
    print(f"  OCR: {safe} -> output/{stem}/ ...", end=" ", flush=True)
    operation = client.async_batch_annotate_files(requests=[request])
    operation.result(timeout=300)
    print("OK")


def main():
    if len(sys.argv) > 1:
        pdfs = [sys.argv[1]]
    else:
        pdfs = ALL_PDFS

    print(f"=== Vision OCR ({len(pdfs)} PDF) ===")
    for pdf in pdfs:
        run_ocr(pdf)
    print("=== Done ===")


if __name__ == "__main__":
    main()
