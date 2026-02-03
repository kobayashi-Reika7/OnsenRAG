# 環境変数 (.env) の使い方

## 1. .env ファイルの記述例

プロジェクト直下（`frontend/` 直下）に `.env` を置きます。

```env
VITE_FIREBASE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789012
VITE_FIREBASE_APP_ID=1:123456789012:web:abcdef123456
```

各値は [Firebase Console](https://console.firebase.google.com/) → プロジェクト設定 → 一般 → アプリ で確認できます。

---

## 2. import.meta.env での読み取り方法

Vite では、**VITE_** で始まる環境変数だけがクライアント（ブラウザ）に公開されます。

```javascript
// 正しい: VITE_ プレフィックス付き
const apiKey = import.meta.env.VITE_FIREBASE_API_KEY;

// 間違い: VITE_ がないと undefined
const secret = import.meta.env.SECRET_KEY;  // → undefined
```

### 注意点

- `.env` を変更したら **`npm run dev` を再起動**すること（ビルド時に読み込まれるため）

---

## 3. .gitignore に .env を追加する理由

| 理由 | 説明 |
|------|------|
| API キー漏洩 | `.env` には Firebase API キーなど秘密情報が含まれる |
| リポジトリ公開時 | GitHub 等に push すると、誰でもキーを閲覧できてしまう |

```gitignore
.env
.env.local
.env.*.local
```
