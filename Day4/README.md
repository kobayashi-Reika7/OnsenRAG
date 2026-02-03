# Day4 ToDoアプリ（Day2相当 + Firestore永続化）

React（Vite）と Firebase Firestore を直結した学習用 ToDo アプリです。  
**Day2 の機能を再現**しつつ、データを Firestore でクラウド永続化します。

- **データ経路**: フロントエンド → Firestore 直接（Firebase SDK）
- **機能**: リスト／期限／お気に入り／カウンター／メモ／タイマー
- **永続化**: リロード・タブを閉じてもデータが残る

## 前提

- React（Vite）
- Firebase v9 以降（modular SDK）
- Firestore（NoSQL）
- 状態管理は useState / useEffect のみ

## React で Firebase を使う準備

詳細な手順は **`frontend/SETUP_FIREBASE.md`** を参照してください。

- 依存関係: `npm install`（Firebase SDK は package.json に含まれています）
- Firebase プロジェクト作成 → Web アプリ追加 → 設定値を控える
- Firestore データベースをテストモードで作成
- `.env.example` を `.env` にコピーし、Firebase の設定値を入れる

**注意（重要）**: `.env` は GitHub に push しない。`.env.example` は push して OK。設定後は `npm run dev` を再起動すること（反映のため）。

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
cd Day4/frontend
npm install
npm run dev
```

ブラウザで **http://localhost:5100** を開く（開発サーバーのポートは 5100 に固定済み）。

## 動作確認（ゴール）

- タスクを追加し、ページをリロードしてもタスクが消えないこと。
- Firebase Console → Firestore で `lists` / `todos` コレクションにドキュメントが増えていること。

**詳細な手順**（確認項目一覧・手順・エラー時の確認）は **`docs/README.md`** の「5. 動作確認」を参照してください。

## ディレクトリ構成（フロントエンド）

```
Day4/frontend/
├── src/
│   ├── firebase/firebase.js      # Firebase 初期化・db の export のみ
│   ├── services/firestore.js     # getLists, addList, deleteList, getTasks, addTask, updateTask, deleteTask
│   ├── utils/taskUtils.js
│   ├── constants/messages.js
│   ├── components/               # ListSelector, TaskForm, TaskList, TaskItem, Counter, Timer, Memo
│   ├── App.jsx
│   └── main.jsx
├── .env.example                  # コピーして .env にリネームし値を入れる
├── package.json
└── vite.config.js
```

## データの流れ

1. **初回表示**: `useEffect` で `getLists()` / `getTasks()` → Firestore から取得 → state 更新 → 画面に表示。
2. **タスク**: 追加は `addTask()`、更新は `updateTask()`、削除は `deleteTask()` で Firestore に反映 → `getTasks()` で再取得して state 更新。
3. **リスト**: 追加は `addList()`、削除は `deleteList()`（属するタスクはデフォルトリストに移動）→ `getLists()` / `getTasks()` で再取得。

## クラウドデプロイ

フロントを Vercel / Netlify へ、バックエンド（任意）を Railway / Render などへデプロイする手順は **`docs/README.md`** の「7. クラウドデプロイ」を参照してください。環境変数（CORS・API ベース URL など）の設定も記載しています。

---

設計・データモデル・コンポーネント・Day2 比較などは **`docs/README.md`**（統合ドキュメント）にまとめています。Day3 由来の `backend/` は参考用です。Day4 の動作には **フロントエンドのみ**（Firestore 接続）が必要です。
