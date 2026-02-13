"""
OnsenRAG プロンプトテンプレート（多言語対応）
================================================
backend/prompts/{lang}/ 配下のテキストファイルからプロンプトを読み込む。

プロンプトの編集は Python コードを触らず .txt ファイルだけで完結する。
lang ディレクトリが見つからない場合はフラット構造にフォールバックし、
それでも見つからない場合は ja にフォールバックする。
"""

import os
from functools import lru_cache

# prompts ディレクトリ（backend/prompts/）
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts"
)

SUPPORTED_LANGS = ("ja", "en")
DEFAULT_LANG = "ja"


@lru_cache(maxsize=None)
def _load(filename: str, lang: str = DEFAULT_LANG, fallback: str = "") -> str:
    """言語別プロンプトを読み込む（キャッシュ付き）。

    探索順:
      1. prompts/{lang}/{filename}
      2. prompts/{filename}          (フラット構造・後方互換)
      3. prompts/{DEFAULT_LANG}/{filename}  (ja へフォールバック)
    """
    candidates = [
        os.path.join(_PROMPTS_DIR, lang, filename),
        os.path.join(_PROMPTS_DIR, filename),
        os.path.join(_PROMPTS_DIR, DEFAULT_LANG, filename),
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue
    return fallback


def load_prompt(filename: str, lang: str = DEFAULT_LANG) -> str:
    """公開API: 言語別プロンプトを取得する。"""
    return _load(filename, lang)


def reload_prompts() -> None:
    """キャッシュをクリアしてプロンプトを再読み込みする"""
    _load.cache_clear()
    # モジュールレベル変数を再設定（ja デフォルト）
    global ANSWER_PROMPT, LLM_EXTRACT_PROMPT, QUERY_RESOLVE_PROMPT
    ANSWER_PROMPT = _load("answer.txt", DEFAULT_LANG)
    LLM_EXTRACT_PROMPT = _load("llm_extract.txt", DEFAULT_LANG)
    QUERY_RESOLVE_PROMPT = _load("query_resolve.txt", DEFAULT_LANG)


# --- モジュールレベルで公開（既存コードとの互換性を維持 — ja デフォルト） ---
ANSWER_PROMPT: str = _load("answer.txt", DEFAULT_LANG)
LLM_EXTRACT_PROMPT: str = _load("llm_extract.txt", DEFAULT_LANG)
QUERY_RESOLVE_PROMPT: str = _load("query_resolve.txt", DEFAULT_LANG)
