"""
草津温泉ガイド RAG 実行スクリプト
data/kusatsu_chunks.json を読み込み
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
# backend/ ディレクトリをパスに追加
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_DIR)

from src.onsen_rag import OnsenRAG

CHUNKS_PATH = [os.path.join(_BACKEND_DIR, "data", "kusatsu_chunks.json")]


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "草津温泉のアクセス方法は？"

    print("=" * 50)
    print("RAG - 草津温泉ガイド（kusatsu_chunks.json）")
    print("=" * 50)

    rag = OnsenRAG(chunk_size=450, chunk_overlap=75)
    rag.load_json_chunks(CHUNKS_PATH)

    print(f"\n[質問] {question}")
    result = rag.query(question)
    print(f"\n[回答] {result['result']}")


if __name__ == "__main__":
    main()
