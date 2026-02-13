"""
社内ドキュメント検索 - 実行スクリプト
======================================

使用例:
  # dataフォルダ内のPDF・TXTを読み込み
  python run_corporate_doc.py

  # 任意のフォルダを指定
  python run_corporate_doc.py /path/to/docs
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
# backend/ ディレクトリをパスに追加
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_DIR)

from _legacy.corporate_doc_rag import CorporateDocRAG


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _BACKEND_DIR, "data"
    )
    question = sys.argv[2] if len(sys.argv) > 2 else "この文書の要点は何ですか？"

    print("=" * 50)
    print("社内ドキュメント検索 - CorporateDocRAG")
    print("=" * 50)

    rag = CorporateDocRAG()
    rag.load_from_folder(folder)

    print(f"\n[質問] {question}")
    result = rag.query(question)
    ans = result["result"]
    if hasattr(ans, "content"):
        ans = str(ans.content)
    print(f"\n[回答] {ans}")


if __name__ == "__main__":
    main()
