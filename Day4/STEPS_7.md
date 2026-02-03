# Firebase + React の 7 ステップ

## 1. Firebase のウェブサイトでプロジェクト作成

- [Firebase Console](https://console.firebase.google.com/) にアクセス
- Google アカウントでログイン → **「プロジェクトを追加」** → プロジェクト名を入力して作成

## 2. Firestore データベースを有効化する

- Firebase Console 左メニュー **「Firestore Database」** → **「データベースを作成」**
- **「テストモードで開始」** を選択 → ロケーション選択 → **「有効にする」**

## 3. React プロジェクトに `npm install firebase`

```bash
cd Day4/frontend
npm install firebase
```

※ すでに `package.json` に firebase が含まれている場合は `npm install` で OK。

## 4. firebase.js を作成し接続設定

- **場所**: `Day4/frontend/src/firebase/firebase.js`
- **内容**: `initializeApp` と `getFirestore` で接続し、`db` を export
- ※ 既に作成済み。設定値は `.env` の `VITE_FIREBASE_*` を参照。

## 5. Cursor で addTodo, getTodos 関数を作成

- **場所**: `Day4/frontend/src/services/firestore.js`
- **関数**: `addTodo(title)` … todos に追加（createdAt 付き）、`getTodos()` … 一覧を配列で返す
- ※ 既に作成済み（`deleteTodo` もあり）。

## 6. React の画面とつなぎこむ（追加 → 表示）

- **App.jsx**: `useEffect` で `getTodos()` 取得 → state に反映。フォームで `addTodo()` 呼び出し。`TodoList` に tasks を渡して表示。
- **TodoList.jsx**: props の tasks を map で一覧表示、削除ボタンで `deleteTodo` 呼び出し。
- ※ 既に接続済み。

## 7. .env を設定して GitHub へプッシュ

**重要**

- **`.env`** … ローカルで Firebase の設定値（`VITE_FIREBASE_*`）を入れる。**GitHub には push しない**（`.gitignore` で除外済み）。
- **`.env.example`** … 値は空のテンプレート。**push して OK**。
- **GitHub へ push するもの** … ソースコード・`.env.example`・`package.json` など（`.env` は含めない）。

`.env` を編集したら `npm run dev` を再起動すること（反映のため）。

---

## チェックリスト

| # | 内容 | 状態 |
|---|------|------|
| 1 | Firebase でプロジェクト作成 | 手動（Console） |
| 2 | Firestore を有効化 | 手動（Console） |
| 3 | `npm install firebase` | 実施済み／要確認時は `npm install` |
| 4 | firebase.js 作成・接続設定 | 済（src/firebase/firebase.js） |
| 5 | addTodo, getTodos 作成 | 済（src/services/firestore.js） |
| 6 | React 画面とつなぎこみ | 済（App.jsx, TodoList.jsx） |
| 7 | .env 設定 ＋ GitHub へ push（.env 除く） | .env はローカルのみ、push 時は .env を含めない |
