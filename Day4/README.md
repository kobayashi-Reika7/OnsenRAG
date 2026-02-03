# Day4 ToDoアプリ（React + Firestore）

React（Vite）と Firebase Firestore を接続した学習用 ToDo アプリです。  
タスクの追加・取得・削除ができ、リロードしてもデータが保持されます。

## 前提

- React（Vite）
- Firebase v9 以降（modular SDK）
- Firestore（NoSQL）
- 状態管理は useState / useEffect のみ

## 環境構築手順

### 1. Node.js

https://nodejs.org/ から LTS をインストールし、`node -v` と `npm -v` で確認する。

### 2. Firebase プロジェクトの準備

1. [Firebase Console](https://console.firebase.google.com/) でプロジェクトを作成する。
2. プロジェクト設定 → 一般 → 「アプリを追加」で Web アプリを追加し、表示される設定（apiKey, authDomain など）を控える。
3. Firestore データベースを作成する（テストモードで開始で可）。

### 3. フロントエンドの環境変数

```bash
cd Day4/frontend
cp .env.example .env
```

`.env` を開き、Firebase の設定値を入れる（VITE_ プレフィックス付きの変数）。

```
VITE_FIREBASE_API_KEY=xxxx
VITE_FIREBASE_AUTH_DOMAIN=xxxx.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=xxxx
VITE_FIREBASE_STORAGE_BUCKET=xxxx.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=xxxx
VITE_FIREBASE_APP_ID=xxxx
```

### 4. 依存関係のインストールと起動

```bash
npm install
npm run dev
```

ブラウザで http://localhost:5173 を開く。

## 動作確認（ゴール）

- タスクを追加し、ページをリロードしてもタスクが消えないこと。
- Firebase Console → Firestore で `todos` コレクションにドキュメントが増えていること。
- React（state）→ Firestore（addTodo/getTodos/deleteTodo）→ React（setTasks）の流れがコード上で追えること。

## ディレクトリ構成（フロントエンド）

```
Day4/frontend/
├── src/
│   ├── firebase/
│   │   └── firebase.js    # Firebase 初期化・db の export のみ
│   ├── services/
│   │   └── firestore.js   # addTodo / getTodos / deleteTodo
│   ├── components/
│   │   └── TodoList.jsx   # タスク一覧表示のみ
│   ├── App.jsx
│   └── main.jsx
├── .env.example           # コピーして .env にリネームし値を入れる
├── package.json
└── vite.config.js
```

## データの流れ

1. **初回表示**: `useEffect` で `getTodos()` → Firestore から取得 → `setTasks` で state 更新 → 画面に表示。
2. **追加**: フォーム送信 → `addTodo(title)` で Firestore に保存 → `getTodos()` で再取得 → state 更新。
3. **削除**: 削除ボタン → `deleteTodo(id)` で Firestore から削除 → `getTodos()` で再取得 → state 更新。

---

Day3 由来の `backend/` や `docs/DESIGN.md` は参考用として残しています。Day4 の動作には **フロントエンドのみ**（Firestore 接続）が必要です。
