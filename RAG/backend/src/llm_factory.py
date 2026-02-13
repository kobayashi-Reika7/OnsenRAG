"""
LLM Factory - LLM初期化の共通モジュール
==========================================
複数のRAGクラスで共有されるLLM初期化ロジックを集約。

レートリミット対策:
  - with_fallbacks() で全モデルをチェーン化
  - 1つのモデルが 429 を返しても自動で次のモデルに切り替え
  - 各モデルが個別クォータを持つため、全滅しにくい

フォールバック順序:
  gemini-2.5-flash-lite → gemini-2.5-flash → gemini-2.5-pro
  → Groq (llama-3.1-8b-instant → gemma2-9b-it → llama-3.3-70b-versatile → mixtral-8x7b-32768)
  → OpenAI
"""

import os
import time
import logging

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def create_llm(temperature: float = 0):
    """
    利用可能な全LLMをフォールバックチェーンで初期化する。

    LangChain の with_fallbacks() を使用し、ランタイムの 429 エラー時に
    自動で次のモデルに切り替わる。

    Returns:
        LangChain LLM インスタンス（フォールバック付き）

    Raises:
        ValueError: 利用可能なLLMがない場合
    """
    google_key = os.getenv("GOOGLE_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    llms = []

    # --- Gemini 複数モデル（各モデルが個別の日別クォータを持つ） ---
    if google_key and not google_key.startswith("your_"):
        gemini_models = [
            "gemini-2.5-flash-lite",   # 最軽量・高速（GA）
            "gemini-2.5-flash",        # 標準 Flash（GA）
            "gemini-2.5-pro",          # 高品質（GA）
        ]
        for model_name in gemini_models:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=google_key,
                    max_retries=1,
                )
                llms.append((llm, f"Gemini ({model_name})"))
            except Exception as e:
                print(f"  [SKIP] Gemini {model_name}: {str(e)[:80]}")

    # --- Groq 複数モデル（各モデルが個別の日別クォータを持つ） ---
    if groq_key and not groq_key.startswith("gsk_your"):
        groq_models = [
            "llama-3.1-8b-instant",        # 軽量・高速・クォータ大
            "gemma2-9b-it",                 # Google Gemma2・日本語対応
            "llama-3.3-70b-versatile",      # 高性能・クォータ小
            "mixtral-8x7b-32768",           # Mixtral MoE
        ]
        for model_name in groq_models:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model=model_name,
                    temperature=temperature,
                    groq_api_key=groq_key,
                )
                llms.append((llm, f"Groq ({model_name})"))
            except Exception as e:
                print(f"  [SKIP] Groq {model_name}: {str(e)[:80]}")

    # --- OpenAI ---
    if openai_key and not openai_key.startswith("sk-your"):
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                temperature=temperature,
                openai_api_key=openai_key,
            )
            llms.append((llm, "OpenAI"))
        except Exception as e:
            print(f"  [SKIP] OpenAI: {str(e)[:80]}")

    if not llms:
        raise ValueError(
            "LLM APIキーが未設定です。\n"
            ".envファイルに GOOGLE_API_KEY, GROQ_API_KEY, "
            "または OPENAI_API_KEY を設定してください。"
        )

    # フォールバックチェーン構築
    primary_llm, primary_name = llms[0]
    fallback_names = []

    if len(llms) > 1:
        fallback_llms = [llm for llm, _ in llms[1:]]
        fallback_names = [name for _, name in llms[1:]]
        primary_llm = primary_llm.with_fallbacks(fallback_llms)

    print(f"  LLM: {primary_name}")
    if fallback_names:
        print(f"  Fallbacks: {' → '.join(fallback_names)}")

    return primary_llm


def invoke_with_retry(chain, inputs: dict, max_retries: int = 3) -> str:
    """
    LLMチェーンを429リトライ付きで実行する。

    with_fallbacks() と組み合わせることで：
      - 各モデル内の一時的な 429 → リトライで回復
      - モデルの日別クォータ枯渇 → fallback で次のモデルへ

    Returns:
        LLMの回答テキスト
    """
    for attempt in range(max_retries):
        try:
            response = chain.invoke(inputs)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = (
                "429" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
                or "rate_limit" in error_str.lower()
            )

            if is_rate_limit and attempt < max_retries - 1:
                wait = min(2 ** attempt * 15, 60)  # 15s, 30s, 60s
                print(f"[RETRY] レートリミット、{wait}秒後にリトライ ({attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
