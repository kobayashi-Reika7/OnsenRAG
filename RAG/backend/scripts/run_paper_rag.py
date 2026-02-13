"""
研究論文アシスタント - 実行スクリプト
======================================

使用例:
  python run_paper_rag.py <PDFパス> "この論文の結論は？"
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
# backend/ ディレクトリをパスに追加
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_DIR)

from _legacy.paper_rag import PaperRAG


def main():
    if len(sys.argv) < 2:
        print("使用例: python run_paper_rag.py <PDFパス> [質問]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "この論文の要点は何ですか？"

    print("=" * 50)
    print("研究論文アシスタント - PaperRAG")
    print("=" * 50)

    rag = PaperRAG()
    rag.load_paper(pdf_path)

    print(f"\n[質問] {question}")
    result = rag.query(question)
    print(f"\n[回答] {result['result']}")
    if result.get("sources"):
        print("\n[出典]")
        for s in result["sources"]:
            print(f"  {s}")


if __name__ == "__main__":
    main()
