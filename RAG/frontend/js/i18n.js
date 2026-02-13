/* ============================================================
   OnsenRAG - i18n (Internationalization) Module
   ============================================================ */

const TRANSLATIONS = {
    ja: {
        title: "OnsenRAG - 温泉の知恵袋",
        badge: "温泉の知恵袋",
        tagline: "温泉のいろは、なんでもおまかせ",
        statusChecking: "接続確認中",
        statusConnected: "接続済み",
        statusInit: "初期化中",
        statusOffline: "オフライン",
        welcomeHeading: "温泉の旅、はじめます",
        welcomeSubtext: "気になることをお気軽にどうぞ",
        example1: "草津温泉の特徴は？",
        example2: "有馬温泉のおすすめは？",
        example3: "別府温泉の泉質は？",
        hint: "質問例をタップするか、下の入力欄に自由に入力できます",
        placeholder: "温泉について質問を入力...",
        send: "送る",
        roleUser: "あなた",
        roleAssistant: "温泉アシスタント",
        loading: "回答を考えています...",
        errorFetch: "回答の取得に失敗しました。",
        errorServer: "APIサーバーが起動していない可能性があります。",
    },
    en: {
        title: "OnsenRAG - Hot Spring Guide",
        badge: "Hot Spring Guide",
        tagline: "Your guide to everything about Japanese hot springs",
        statusChecking: "Checking connection",
        statusConnected: "Connected",
        statusInit: "Initializing",
        statusOffline: "Offline",
        welcomeHeading: "Your hot spring journey starts here",
        welcomeSubtext: "Feel free to ask anything",
        example1: "What are the features of Kusatsu Onsen?",
        example2: "What do you recommend in Arima Onsen?",
        example3: "What is the water quality of Beppu Onsen?",
        hint: "Tap an example or type your question below",
        placeholder: "Ask about hot springs...",
        send: "Send",
        roleUser: "You",
        roleAssistant: "Onsen Assistant",
        loading: "Thinking...",
        errorFetch: "Failed to get a response.",
        errorServer: "The API server may not be running.",
    }
};

// --- 言語管理 ---
const I18N_STORAGE_KEY = "onsenrag_lang";

function getDefaultLang() {
    const stored = localStorage.getItem(I18N_STORAGE_KEY);
    if (stored && TRANSLATIONS[stored]) return stored;
    const browserLang = (navigator.language || "ja").slice(0, 2);
    return TRANSLATIONS[browserLang] ? browserLang : "ja";
}

let currentLang = getDefaultLang();

function getCurrentLang() {
    return currentLang;
}

function setLang(lang) {
    if (!TRANSLATIONS[lang]) return;
    currentLang = lang;
    localStorage.setItem(I18N_STORAGE_KEY, lang);
    applyTranslations();
    document.documentElement.lang = lang;
    // 言語切替ボタンの表示を更新
    const btn = document.getElementById("langToggle");
    if (btn) btn.textContent = lang === "ja" ? "EN" : "JA";
}

function t(key) {
    const dict = TRANSLATIONS[currentLang] || TRANSLATIONS["ja"];
    return dict[key] || TRANSLATIONS["ja"][key] || key;
}

function applyTranslations() {
    // data-i18n 属性を持つ要素にテキストを適用
    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        el.textContent = t(key);
    });
    // data-i18n-placeholder 属性を持つ要素
    document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
        const key = el.getAttribute("data-i18n-placeholder");
        el.placeholder = t(key);
    });
    // タイトル更新
    document.title = t("title");
}

function toggleLang() {
    setLang(currentLang === "ja" ? "en" : "ja");
}
