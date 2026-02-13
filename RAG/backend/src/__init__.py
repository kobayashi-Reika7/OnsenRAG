# ===========================================
# OnsenRAG - RAGシステムパッケージ
# ===========================================
# 温泉特化RAGシステムのコアモジュール群。
#
# モジュール構成:
#   config.py          - 定数・パス・パラメータ
#   prompts.py         - プロンプトテンプレート
#   data_loader.py     - データ読み込み
#   search_pipeline.py - 検索パイプライン
#   onsen_rag.py       - メインRAGクラス
#   support_bot.py     - カスタマーサポートボット
#   llm_factory.py     - LLM初期化ファクトリ
#   text_splitter_utils.py - テキスト分割ユーティリティ

from src.onsen_rag import OnsenRAG
from src.support_bot import SupportBot, SupportResponse

__all__ = [
    "OnsenRAG",
    "SupportBot",
    "SupportResponse",
]
