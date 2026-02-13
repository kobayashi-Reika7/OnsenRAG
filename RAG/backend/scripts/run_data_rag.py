"""
RAG/data をインプットとするRAG実行スクリプト
============================================

dataフォルダ内のPDF・TXTを読み込み、質問に回答します。

使用例:
  python run_data_rag.py "温泉について教えてください"
  python run_data_rag.py                    # デフォルト質問
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
# backend/ ディレクトリをパスに追加
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_DIR)

from _legacy.corporate_doc_rag import CorporateDocRAG

# backend/data をインプットとして使用
DATA_DIR = os.path.join(_BACKEND_DIR, "data")


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "この文書の要点は何ですか？"

    print("=" * 50)
    print("RAG - data フォルダをインプット")
    print("=" * 50)

    rag = CorporateDocRAG()
    rag.load_from_folder(DATA_DIR)

    print(f"\n[質問] {question}")
    result = rag.query(question)
    ans = result["result"]
    if hasattr(ans, "content"):
        ans = str(ans.content)
    print(f"\n[回答] {ans}")


if __name__ == "__main__":
    main()
