"""
最小動作確認スクリプト（APIキー不要）
=======================================
Embedding + ChromaDB の検索部分のみを確認するスクリプト。
LLM（OpenAI）の回答生成は行わないので、APIキーなしで実行可能。

実行方法:
  python test_search.py

確認できること:
  - Embeddingモデルが正しくダウンロード・動作するか
  - テキストがチャンクに分割されるか
  - ChromaDBに保存されるか
  - 質問に対して正しいチャンクが検索されるか
"""

import os
import sys

# backend/ ディレクトリをパスに追加
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_DIR)

from src.text_splitter_utils import create_token_text_splitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


def main():
    print("=" * 60)
    print("♨️ OnsenRAG 最小動作確認（APIキー不要）")
    print("=" * 60)

    # ステップ1: 温泉テキストデータを読み込み
    data_path = os.path.join(_BACKEND_DIR, "data", "onsen_knowledge.txt")

    if not os.path.exists(data_path):
        print(f"❌ データファイルが見つかりません: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n📄 ステップ1: テキスト読み込み完了 ({len(text)}文字)")

    # ステップ2: テキストをチャンク化
    text_splitter = create_token_text_splitter(
        chunk_size=450,
        chunk_overlap=75
        separators=["■ ", "\n\n", "\n", "。", "、", " ", ""]
    )
    document = Document(page_content=text)
    splits = text_splitter.split_documents([document])

    print(f"✂️ ステップ2: {len(splits)}件のチャンクに分割")
    for i, split in enumerate(splits):
        preview = split.page_content[:50].replace("\n", " ")
        print(f"  [{i+1}] {preview}...")

    # ステップ3: Embeddingモデルでベクトル化 → ChromaDBに保存
    print("\n⏳ ステップ3: Embeddingモデルをロード中...")
    print("  （初回は模型のダウンロードに数分かかります）")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    print("✅ ステップ3: ベクトルDBに保存完了")

    # ステップ4: 質問に対して検索テスト
    print("\n" + "=" * 60)
    print("🔍 ステップ4: 検索テスト")
    print("=" * 60)

    test_questions = [
        "温泉とは何ですか？",
        "美肌の湯と呼ばれる泉質は？",
        "冬におすすめの温泉地は？",
        "東京から行きやすい温泉地は？",
        "刺激が少ない温泉はどれですか？",
        "湯めぐりを楽しめる温泉地は？",
    ]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    for question in test_questions:
        print(f"\n❓ 質問: {question}")
        docs = retriever.invoke(question)
        for i, doc in enumerate(docs):
            content = doc.page_content.replace("\n", " ")
            preview = content[:80] + "..." if len(content) > 80 else content
            print(f"  [{i+1}] {preview}")

    # 結果サマリー
    print("\n" + "=" * 60)
    print("✅ 動作確認完了！")
    print("=" * 60)
    print()
    print("確認できたこと:")
    print("  ✅ テキストの読み込み・チャンク分割")
    print("  ✅ Embeddingモデル（multilingual-e5-base）の動作")
    print("  ✅ ChromaDB へのベクトル保存")
    print("  ✅ 質問に対する類似チャンク検索")
    print()
    print("次のステップ:")
    print("  1. .envファイルにOpenAI APIキーを設定")
    print("     OPENAI_API_KEY=sk-your-actual-key")
    print("  2. python main.py で完全なRAGデモを実行")
    print("  3. uvicorn api.main:app --reload --port 8000 でチャットAPI起動")


if __name__ == "__main__":
    main()
