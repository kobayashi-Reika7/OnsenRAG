"""
OnsenRAG - メインエントリポイント
==================================
RAGシステムの使用例を示すスクリプト。
各クラスの基本的な使い方をデモンストレーションする。

実行方法:
  python backend/main.py

前提条件:
  - .envファイルにAPIキーを設定済み
  - pip install -r requirements.txt でライブラリインストール済み
"""

import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()


def demo_onsen_rag():
    """
    OnsenRAGのデモ（メイン）

    温泉テキストデータ（data/onsen_knowledge.txt）を読み込み、
    サンプル質問で検索・回答を行う。
    """
    from src.onsen_rag import OnsenRAG

    print("=" * 60)
    print("OnsenRAG Demo (onsen_knowledge.txt)")
    print("=" * 60)

    # OnsenRAGの初期化（chunk_size=450 tokens、overlap=75 tokens）
    rag = OnsenRAG(chunk_size=450, chunk_overlap=75)
    rag.load_data()

    # サンプル質問で回答を確認
    sample_questions = [
        "温泉とは何ですか？",
        "美肌の湯と呼ばれる泉質は？",
        "冬におすすめの温泉地は？",
        "東京から行きやすい温泉地は？",
    ]

    for question in sample_questions:
        print(f"\n[Q] {question}")
        result = rag.query(question)
        print(f"[A] {result['result'].strip()}")
        print("-" * 40)

    # 一括評価
    print("\n[EVAL] Running batch evaluation...")
    rag.evaluate()


def demo_onsen_rag_json():
    """
    OnsenRAGのデモ（JSONチャンク使用）

    草津・箱根のJSONチャンクをVector DBに格納し、質問に回答する。
    """
    from src.onsen_rag import OnsenRAG, DEFAULT_JSON_CHUNK_PATHS

    print("=" * 60)
    print("♨️ OnsenRAG デモ（JSONチャンク・草津+箱根）")
    print("=" * 60)

    rag = OnsenRAG(chunk_size=450, chunk_overlap=75)
    rag.load_json_chunks(DEFAULT_JSON_CHUNK_PATHS)

    sample_questions = [
        "草津温泉へのアクセス方法を教えてください",
        "湯畑のライトアップ時間は？",
        "箱根町総合観光案内所の電話番号は？",
        "別府八湯とは何ですか？",
        "有馬温泉の公衆浴場はどこですか？",
    ]

    for question in sample_questions:
        print(f"\n❓ 質問: {question}")
        result = rag.query(question, k=3)
        print(f"💡 回答: {result['result'].strip()}")
        print("-" * 40)


def demo_simple_rag():
    """
    SimpleRAGのデモ（レガシー）

    テキストデータからRAGシステムを構築し、質問に回答する例。
    温泉に関するサンプルデータを使用。
    """
    from _legacy.simple_rag import SimpleRAG

    print("=" * 60)
    print("📚 SimpleRAG デモ")
    print("=" * 60)

    # サンプルデータ（温泉に関する情報）
    sample_texts = [
        "草津温泉は群馬県にある日本三名泉の一つです。"
        "自然湧出量は日本一で、毎分3万2300リットル以上の温泉が湧き出ています。"
        "泉質は酸性・含硫黄の温泉で、殺菌力が非常に強いのが特徴です。",

        "別府温泉は大分県別府市にある温泉群で、源泉数・湧出量ともに日本一です。"
        "地獄めぐりが有名で、海地獄、血の池地獄、龍巻地獄などの観光名所があります。"
        "泉質は多彩で、単純温泉、炭酸水素塩泉、硫黄泉など11種類中10種類が存在します。",

        "下呂温泉は岐阜県にある日本三名泉の一つです。"
        "アルカリ性単純温泉で、pH値9.18のなめらかな湯ざわりが特徴です。"
        "美肌の湯として知られ、肌がすべすべになると評判です。",

        "有馬温泉は兵庫県神戸市にある日本三古泉の一つです。"
        "金泉（鉄分を含む赤茶色の湯）と銀泉（炭酸泉・ラジウム泉）の2種類があります。"
        "太閤秀吉が愛した温泉としても有名です。",
    ]

    # RAGシステムの初期化とデータ読み込み
    rag = SimpleRAG()
    rag.load_documents(sample_texts)

    # 質問と回答
    question = "日本三名泉はどこですか？"
    print(f"\n❓ 質問: {question}")

    result = rag.query(question)
    print(f"💡 回答: {result['result']}")
    print(f"📄 参照ドキュメント数: {len(result['source_documents'])}")


def demo_pdf_rag():
    """
    JSONチャンクRAGのデモ（草津温泉）
    data/kusatsu_chunks.json から読み込み
    """
    from src.onsen_rag import OnsenRAG

    print("\n" + "=" * 60)
    print("RAG デモ（草津温泉 chunks）")
    print("=" * 60)

    chunks_path = os.path.join(os.path.dirname(__file__), "data", "kusatsu_chunks.json")
    if not os.path.exists(chunks_path):
        print(f"チャンクファイルが見つかりません: {chunks_path}")
        return

    rag = OnsenRAG(chunk_size=450, chunk_overlap=75)
    rag.load_json_chunks([chunks_path])

    question = "草津温泉のアクセス方法は？"
    print(f"\n[Q] {question}")
    result = rag.query(question)
    print(f"[A] {result['result']}")


def demo_hybrid_search():
    """
    HybridSearchRAGのデモ

    セマンティック検索とキーワード検索を組み合わせた
    ハイブリッド検索の精度を確認する例。
    """
    from _legacy.hybrid_search_rag import HybridSearchRAG

    print("\n" + "=" * 60)
    print("🔀 HybridSearchRAG デモ")
    print("=" * 60)

    # サンプルデータ
    sample_texts = [
        "草津温泉は群馬県にある日本三名泉の一つです。"
        "自然湧出量は日本一で、泉質は酸性・含硫黄の温泉です。"
        "湯畑は草津温泉のシンボルで、毎分4000リットルの温泉が湧き出ています。",

        "別府温泉は大分県別府市にある温泉群で、源泉数・湧出量ともに日本一です。"
        "地獄めぐりが有名な観光名所です。",

        "温泉の効能には、神経痛、筋肉痛、関節痛、冷え性、疲労回復などがあります。"
        "泉質によって効能が異なるため、目的に合った温泉を選ぶことが大切です。",

        "露天風呂は日本の温泉文化の象徴です。"
        "自然の中で入浴を楽しめるため、リラックス効果が高いとされています。",
    ]

    # セマンティック検索を重視（重み0.7）
    rag = HybridSearchRAG(semantic_weight=0.7)
    rag.load_documents(sample_texts)

    # 検索のみ（デバッグ用）
    question = "草津温泉の特徴"
    rag.search_only(question, k=3)


def demo_reranking():
    """
    ReRankingRAGのデモ

    初期検索→Re-ranking→回答生成の流れを確認する例。
    """
    from _legacy.reranking_rag import ReRankingRAG

    print("\n" + "=" * 60)
    print("🏆 ReRankingRAG デモ")
    print("=" * 60)

    # サンプルデータ
    sample_texts = [
        "草津温泉は群馬県にある日本三名泉の一つです。"
        "自然湧出量は日本一で、泉質は酸性・含硫黄の温泉です。",

        "別府温泉は大分県にある温泉群で、源泉数が日本一です。",

        "下呂温泉は岐阜県にあり、美肌の湯として知られています。",

        "有馬温泉は兵庫県にあり、金泉と銀泉の2種類の温泉があります。",

        "温泉の入り方には、かけ湯、半身浴、全身浴などのマナーがあります。",
    ]

    rag = ReRankingRAG()
    rag.load_documents(sample_texts)

    # Re-ranking付きで回答
    question = "酸性の温泉はどこですか？"
    print(f"\n❓ 質問: {question}")

    result = rag.query(question, initial_k=5, final_k=2)
    print(f"\n💡 回答: {result['result']}")


def main():
    """
    メイン関数

    APIキーの確認後、各デモを順番に実行する。
    使いたいデモのコメントアウトを外して実行してください。
    """
    # APIキーの確認（Gemini / Groq / OpenAI）
    google_key = os.getenv("GOOGLE_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    has_google = google_key and not google_key.startswith("your_")
    has_groq = groq_key and not groq_key.startswith("gsk_your")
    has_openai = openai_key and not openai_key.startswith("sk-your")

    if not has_google and not has_groq and not has_openai:
        print("[WARN] LLM API key is not set.")
        print("  .envファイルに以下のいずれかを設定してください:")
        print("  GOOGLE_API_KEY=AIza...（Gemini・無料・推奨）")
        print("  GROQ_API_KEY=gsk_...")
        print("  OPENAI_API_KEY=sk-...")
        return

    print("[START] OnsenRAG demo\n")

    # 温泉テキストデータを使ったデモ（メイン）
    demo_onsen_rag()

    # JSONチャンク（草津温泉ガイド）を使ったデモ
    # demo_onsen_rag_json()

    # その他のデモ（使いたいもののコメントアウトを外してください）
    # demo_simple_rag()
    # demo_pdf_rag()
    # demo_hybrid_search()
    # demo_reranking()

    print("\n" + "=" * 60)
    print("[OK] Demo completed")
    print("=" * 60)


# スクリプトとして直接実行された場合のみmain()を呼ぶ
if __name__ == "__main__":
    main()
