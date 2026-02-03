# Cursor に使う Day4 専用プロンプト（コピペ用）

Firebase のコードは複雑なので、AI に書いてもらうのが一番です。以下を Cursor のチャットにコピペして使ってください。

---

## プロンプト（そのままコピペ）

```
Day4 の React + Firebase（Firestore）の実装をしてください。

【設定ファイル作成】
React で Firebase を使うために src/firebase/firebase.js を作ってください。
initializeApp と getFirestore を設定し、db をエクスポートしてください。

【追加処理 (Create)】
Firestore の "todos" コレクションにデータを追加する addTodoToDB 関数を作成してください。
引数でタイトルを受け取り、createdAt も保存してください。

【一覧取得 (Read)】
Firestore の "todos" コレクションを取得する getTodosFromDB 関数を作成してください。
React の state に反映しやすい配列の形で返してください。

【環境変数 (.env)】
Firebase の設定値を .env ファイルに分離し、import.meta.env で読み取る方法を教えてください。
また、.gitignore の設定方法も教えてください。
```

---

## 環境変数 (.env) と .gitignore の要点（回答例）

AI に聞かずに自分で設定する場合の要点です。

### .env で設定値を分離する

1. **`Day4/frontend/.env`** に Firebase の設定を書く（このファイルは GitHub に push しない）。

   ```
   VITE_FIREBASE_API_KEY=xxxx
   VITE_FIREBASE_AUTH_DOMAIN=xxxx.firebaseapp.com
   VITE_FIREBASE_PROJECT_ID=xxxx
   VITE_FIREBASE_STORAGE_BUCKET=xxxx.appspot.com
   VITE_FIREBASE_MESSAGING_SENDER_ID=xxxx
   VITE_FIREBASE_APP_ID=xxxx
   ```

2. **Vite では `VITE_` プレフィックス付きの変数のみ** クライアントに公開される。  
   `src/firebase/firebase.js` で次のように読む。

   ```javascript
   const firebaseConfig = {
     apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
     authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
     projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
     storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
     messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
     appId: import.meta.env.VITE_FIREBASE_APP_ID,
   };
   ```

3. **`.env` を変更したら `npm run dev` を再起動する**（反映のため）。

### .gitignore の設定

**`Day4/frontend/.gitignore`** に次を追加する（すでにあればそのままで OK）。

```
.env
.env.local
```

- **`.env`** … GitHub に push しない（API キー等が含まれるため）。
- **`.env.example`** … 値は空のテンプレートなので push して OK。

---

## 現在の実装との対応

| プロンプトで書く名前 | 既存コードの名前 | 備考 |
|----------------------|------------------|------|
| addTodoToDB          | addTodo           | 同じ処理（todos に title, createdAt を保存） |
| getTodosFromDB       | getTodos         | 同じ処理（配列で返す） |

AI に「addTodoToDB」「getTodosFromDB」で書いてもらった場合、既存の `App.jsx` で `addTodo` / `getTodos` を import しているなら、  
`services/firestore.js` で `addTodoToDB` を `addTodo` として export するか、`App.jsx` の import を `addTodoToDB` / `getTodosFromDB` に合わせて変更すればよい。
