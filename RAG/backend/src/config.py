"""
OnsenRAG 設定・定数（多言語対応）
==================================
パス、モデル名、検索パラメータなどの共通設定を一元管理する。
"""

import os

from dotenv import load_dotenv

# HuggingFaceモデルのリモート確認をスキップ（キャッシュ済みなら高速起動）
# ※ 他モジュールの import より前に実行されるよう config.py に配置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

load_dotenv()

# ============================================================
# 言語設定
# ============================================================
SUPPORTED_LANGS = ("ja", "en")
DEFAULT_LANG = "ja"

# ============================================================
# パス定数
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# チャンク JSON ファイル名リスト（言語非依存）
JSON_CHUNK_FILES = [
    "kusatsu_chunks.json",
    "hakone_chunks.json",
    "beppu_chunks.json",
    "arima_chunks.json",
    "onsen_knowledge_chunks.json",
]


def get_data_dir(lang: str = DEFAULT_LANG) -> str:
    """言語別データディレクトリを返す。サブフォルダが無ければフラットにフォールバック。"""
    lang_dir = os.path.join(DATA_DIR, lang)
    if os.path.isdir(lang_dir):
        return lang_dir
    return DATA_DIR


def get_json_chunk_paths(lang: str = DEFAULT_LANG) -> list[str]:
    """言語別の JSON チャンクファイルパスリストを返す。"""
    data_dir = get_data_dir(lang)
    return [os.path.join(data_dir, f) for f in JSON_CHUNK_FILES]


def get_questions_path(lang: str = DEFAULT_LANG) -> str:
    """言語別の sample_questions.json パスを返す。"""
    return os.path.join(get_data_dir(lang), "sample_questions.json")


# --- 後方互換: 既存コードから参照されるモジュールレベル定数（ja デフォルト） ---
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "onsen_knowledge.txt")
DEFAULT_QUESTIONS_PATH = get_questions_path(DEFAULT_LANG)
DEFAULT_KUSATSU_CHUNKS_PATH = os.path.join(get_data_dir(DEFAULT_LANG), "kusatsu_chunks.json")
DEFAULT_JSON_CHUNK_PATHS = get_json_chunk_paths(DEFAULT_LANG)
DEFAULT_TXT_PATHS: list[str] = []

# ChromaDB永続化先（言語別に分離）
_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))


def get_chroma_persist_dir(lang: str = DEFAULT_LANG) -> str:
    """言語別 ChromaDB 永続化ディレクトリ。"""
    return os.path.join(_BACKEND_DIR, f"chroma_onsen_db_{lang}")


def get_chroma_hash_file(lang: str = DEFAULT_LANG) -> str:
    """言語別 ChromaDB ハッシュファイル。"""
    return os.path.join(get_chroma_persist_dir(lang), "_data_hash.txt")


CHROMA_PERSIST_DIR = get_chroma_persist_dir(DEFAULT_LANG)
CHROMA_HASH_FILE = get_chroma_hash_file(DEFAULT_LANG)

# ============================================================
# 温泉地名 → chunk_id プレフィックス対応表（言語別）
# ============================================================
LOCATION_KEYWORDS_BY_LANG: dict[str, dict[str, list[str]]] = {
    "ja": {
        "kusatsu": ["草津"],
        "hakone": ["箱根"],
        "beppu": ["別府"],
        "arima": ["有馬"],
    },
    "en": {
        "kusatsu": ["Kusatsu"],
        "hakone": ["Hakone"],
        "beppu": ["Beppu"],
        "arima": ["Arima"],
    },
}

# 後方互換: 既存コードから LOCATION_KEYWORDS として参照される
LOCATION_KEYWORDS: dict[str, list[str]] = LOCATION_KEYWORDS_BY_LANG[DEFAULT_LANG]


def get_location_keywords(lang: str = DEFAULT_LANG) -> dict[str, list[str]]:
    """言語別の温泉地名キーワード辞書を返す。"""
    return LOCATION_KEYWORDS_BY_LANG.get(lang, LOCATION_KEYWORDS_BY_LANG[DEFAULT_LANG])


# ============================================================
# モデル設定
# ============================================================
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# ============================================================
# 検索パラメータ
# ============================================================
CONFIDENCE_THRESHOLD = -3.0  # CrossEncoderスコアの信頼度閾値

# クエリキャッシュ設定
QUERY_CACHE_MAXSIZE = 128
QUERY_CACHE_TTL = 300  # 5分
