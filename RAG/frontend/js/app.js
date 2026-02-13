/* ============================================================
   OnsenRAG - Hot Spring Guide Application JS (Bilingual)
   ============================================================ */

// APIのベースURL
const API_BASE = (function () {
    const host = window.location.hostname;
    if (!host || host === "localhost" || host === "127.0.0.1") {
        return "http://localhost:8000";
    }
    return window.location.origin;
})();

// --- セッションID（Firestore保存用） ---
const SESSION_ID = crypto.randomUUID();

// --- DOM要素 ---
const chatArea = document.getElementById("chatArea");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("statusText");

let isFirstMessage = true;
const chatHistory = []; // 会話履歴（{role, content}）

// --- 初期化時に i18n を適用 ---
(function initI18n() {
    // i18n.js がロード済みなら初期適用
    if (typeof applyTranslations === "function") {
        applyTranslations();
    }
})();

// --- ヘルスチェック ---
let statusType = "Checking";
(async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        // 現在の言語の RAG が初期化済みか確認
        const lang = typeof getCurrentLang === "function" ? getCurrentLang() : "ja";
        const langReady = data.languages && data.languages[lang];
        if (langReady) {
            statusEl.classList.add("connected");
            statusType = "Connected";
        } else if (data.languages && Object.values(data.languages).some(v => v)) {
            statusEl.classList.add("connected");
            statusType = "Connected";
        } else {
            statusType = "Init";
        }
    } catch {
        statusType = "Offline";
    }
    const statusKey = "status" + statusType;
    statusText.textContent = typeof t === "function" ? t(statusKey) : statusType;
})();

// --- メッセージ表示 ---
function addMessage(text, role, isError = false) {
    if (isFirstMessage) {
        chatArea.innerHTML = "";
        isFirstMessage = false;
    }

    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    const roleName = typeof t === "function"
        ? (role === "user" ? t("roleUser") : t("roleAssistant"))
        : (role === "user" ? "You" : "Assistant");
    meta.innerHTML = `<span class="message-role">${roleName}</span>`;

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${isError ? "error-bubble" : ""}`;
    bubble.textContent = text;

    messageDiv.appendChild(meta);
    messageDiv.appendChild(bubble);

    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

// --- ローディング表示 ---
function setLoading(show) {
    const existing = document.querySelector(".loading");
    if (existing) existing.remove();

    if (show) {
        const loadingText = typeof t === "function" ? t("loading") : "Thinking...";
        const loading = document.createElement("div");
        loading.className = "loading";
        loading.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span></div>${loadingText}`;
        chatArea.appendChild(loading);
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

// --- 質問例ボタン ---
function useExample(btn) {
    // Get text from the inner span (data-i18n element), not the emoji icon
    const spans = btn.querySelectorAll("span");
    const textSpan = spans.length > 1 ? spans[spans.length - 1] : spans[0];
    const text = textSpan ? textSpan.textContent.trim() : btn.textContent.trim();
    questionInput.value = text;
    askQuestion();
}

// --- チャンク参照の除去（日英両対応） ---
function stripChunkRefs(text) {
    // 根拠チャンクID / 参照ソース / 出典 / Sources / References 等のヘッダー以降を全削除
    text = text.split(/\n*[#*_\-\s]*(?:根拠チャンク\s*ID|根拠チャンク|参照チャンク\s*ID|参照チャンク|参照ソース|参照元|参考情報|参考文献|出典|引用元|Sources?|References?|Citations?|chunk_id|source|reference)[*_\s]*[：:]/i)[0] || text;
    // インライン参照 (arima_001), [kusatsu_002] 等を除去
    text = text.replace(/[\(（\[]\s*(?:arima|kusatsu|hakone|beppu|onsen_knowledge)_\d+(?:\s*[,、]\s*(?:arima|kusatsu|hakone|beppu|onsen_knowledge)_\d+)*\s*[\)）\]]/g, '');
    // 行頭の "根拠チャンクID:" / "Sources:" 等を含む行全体を除去
    text = text.replace(/^.*(?:根拠チャンク|参照ソース|参照元|出典|Sources?|References?|Citations?|chunk_id).*$/gim, '');
    return text.replace(/\n{3,}/g, '\n\n').trim();
}

// --- 質問送信 ---
async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    const lang = typeof getCurrentLang === "function" ? getCurrentLang() : "ja";

    questionInput.value = "";
    askBtn.disabled = true;
    addMessage(question, "user");
    chatHistory.push({ role: "user", content: question });
    setLoading(true);

    try {
        // 直近4件の履歴を送信（現在の質問を除く）
        const recentHistory = chatHistory.slice(-5, -1);
        const response = await fetch(`${API_BASE}/api/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                history: recentHistory,
                session_id: SESSION_ID,
                lang: lang,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "API Error");
        }

        const data = await response.json();
        setLoading(false);
        const cleanAnswer = stripChunkRefs(data.answer);
        addMessage(cleanAnswer, "assistant");
        chatHistory.push({ role: "assistant", content: cleanAnswer });
    } catch (error) {
        setLoading(false);
        const errFetch = typeof t === "function" ? t("errorFetch") : "Failed to get a response.";
        const errServer = typeof t === "function" ? t("errorServer") : "The API server may not be running.";
        let errorMsg = errFetch;
        if (error.message.includes("Failed to fetch")) {
            errorMsg += `\n${errServer}\n→ uvicorn backend.api.main:app --reload --port 8000`;
        } else {
            errorMsg += `\n${error.message}`;
        }
        addMessage(errorMsg, "assistant", true);
    }

    askBtn.disabled = false;
    questionInput.focus();
}
