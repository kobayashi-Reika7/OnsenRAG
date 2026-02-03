# Day3 ToDoアプリ 設計書（React + FastAPI）

## 1. 全体アーキテクチャ（文章説明）

```
[ブラウザ]
    │
    │  HTTP (fetch) / JSON
    ▼
[React フロントエンド]  ←  Vite 開発サーバ (例: localhost:5173)
    │
    │  ・画面表示（コンポーネント）
    │  ・ユーザー操作の受付
    │  ・API 呼び出し（fetch）
    │  ・最小限の状態（useState）
    │
    │  GET/POST/PUT/DELETE  (JSON)
    ▼
[FastAPI バックエンド]  ←  uvicorn (例: localhost:8000)
    │
    │  ・ルーティング（/tasks, /lists）
    │  ・データの永続化（メモリ or SQLite）
    │  ・CORS 許可（フロントのオリジン）
    │
    ▼
[データ]  ※ メモリ内のリスト（再起動でリセット） or SQLite
```

- **フロント**と**バック**は別プロセスで動かす（別ポート）。
- フロントは「表示と入力」、バックは「データの管理と整合性」を担当する。
- 認証は行わず、学習用にシンプルに保つ。

---

## 2. ディレクトリ構成例

```
Day3/
├── docs/
│   └── DESIGN.md          # 本設計書
├── backend/               # FastAPI
│   ├── main.py            # アプリ起動・ルート・CORS
│   ├── models.py          # Pydantic モデル（Task, List）
│   ├── store.py           # メモリ上のデータ保持
│   └── requirements.txt
├── frontend/              # React (Vite)
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── api.js         # fetch でバックエンド呼び出し
│   │   ├── components/
│   │   │   ├── TaskList.jsx
│   │   │   ├── TaskItem.jsx
│   │   │   ├── TaskForm.jsx
│   │   │   ├── Counter.jsx
│   │   │   ├── ListSelector.jsx
│   │   │   ├── Timer.jsx
│   │   │   └── Memo.jsx
│   │   └── App.css
│   └── ...
├── todo_requirements.md   # Day2 由来の要件（参考）
└── README.md              # 環境構築・起動手順
```

---

## 3. 主要コンポーネントの役割

| コンポーネント | 責務 |
|----------------|------|
| **App** | 全体の状態（tasks, lists, currentListId）の保持。API から取得・更新。子コンポーネントへ props で渡す。 |
| **TaskList** | 表示対象タスク一覧を受け取り、各 TaskItem を並べて表示する。 |
| **TaskItem** | 1 タスクの表示・完了トグル・編集・削除・お気に入り・期限表示。内部に Timer と Memo を含む。 |
| **TaskForm** | 入力欄と「追加」ボタン。送信時に親の addTask（API 呼び出し）を実行。 |
| **Counter** | 未完了・完了・お気に入り・期限切れの件数を受け取り表示するだけ（表示専用）。 |
| **ListSelector** | リスト一覧の取得・選択・リスト追加・削除。選択変更時に親の setCurrentList を呼ぶ。 |
| **Timer** | 1 タスク用の経過時間表示・開始・停止・リセット。※ 経過秒数はフロントで計算し、必要なら PUT で time を送る。 |
| **Memo** | 1 タスク用のテキストエリア。入力変更を debounce または onBlur で親に渡し、PUT で保存。 |

※ 1 コンポーネント = 1 責務を意識する。

---

## 4. API 設計一覧

| メソッド | パス | 説明 | リクエスト例 | レスポンス例 |
|----------|------|------|--------------|--------------|
| GET | /tasks | タスク一覧取得 | - | `[{ id, title, isCompleted, ... }]` |
| POST | /tasks | タスク追加 | `{ title, listId }` | `{ id, title, ... }` |
| PUT | /tasks/{id} | タスク更新 | `{ title?, isCompleted?, ... }` | 更新後のタスク |
| DELETE | /tasks/{id} | タスク削除 | - | 204 No Content |
| GET | /lists | リスト一覧取得 | - | `[{ id, name }]` |
| POST | /lists | リスト追加 | `{ name }` | `{ id, name }` |

- すべて JSON。CORS はバックエンドでフロントのオリジン（例: http://localhost:5173）を許可する。

---

## 5. 環境構築手順

### フロントエンド（React）

1. **Node.js のインストール**  
   https://nodejs.org/ から LTS をインストール。`node -v` と `npm -v` で確認。
2. **React プロジェクト作成（Vite）**  
   `cd Day3` の上で `npm create vite@latest frontend -- --template react` を実行。`frontend` にプロジェクトができる。
3. **依存関係インストール**  
   `cd frontend` の上で `npm install`。
4. **開発サーバー起動**  
   `npm run dev`。表示された URL（例: http://localhost:5173）で開く。

### バックエンド（FastAPI）

1. **Python**  
   Python 3.9 以上をインストール。`python -V` で確認。
2. **仮想環境作成**  
   `cd Day3/backend` の上で `python -m venv venv`。
3. **有効化**  
   - Windows: `venv\Scripts\activate`  
   - macOS/Linux: `source venv/bin/activate`
4. **FastAPI 等インストール**  
   `pip install fastapi uvicorn`（または `pip install -r requirements.txt`）。
5. **サーバー起動**  
   `uvicorn main:app --reload --host 0.0.0.0 --port 8000`。  
   API は http://localhost:8000 で動作。ドキュメントは http://localhost:8000/docs。

### 連携の注意

- 先にバックエンドを起動してから、フロントの `npm run dev` で開く。
- フロントの `api.js` のベース URL を `http://localhost:8000` に合わせる。

---

## 6. 強み・弱みの考察

### React を使った場合

- **強み**  
  - コンポーネント単位で分割でき、責務が分かりやすい。  
  - 状態が変わった部分だけ再描画されるイメージを持ちやすい。  
  - Hooks（useState, useEffect）で「状態」と「副作用」を一箇所にまとめられる。
- **弱み**  
  - 環境構築（Node, npm, Vite）が必要。  
  - 素の HTML/JS より概念が多く、最初は「どこに何を書くか」で迷いやすい。
- **素の JavaScript との違い**  
  - 素の JS は「DOM を直接触って更新」が中心。React は「状態（state）を更新すると、それに合わせて UI が決まる」という宣言的な書き方。  
  - コンポーネントの再利用や、親子間のデータの流れ（props）が明確になる。

### FastAPI を使った場合

- **強み**  
  - 型（Pydantic）でリクエスト・レスポンスを定義できる。  
  - `/docs` で API を試せて学習しやすい。  
  - 非同期対応しやすく、将来 DB や外部 API を足しやすい。
- **弱み**  
  - Python の知識と、HTTP/API の考え方が必要。  
  - フロントと別プロセスなので、「どこでエラーが出ているか」の切り分けが必要。
- **フロントと分離するメリット**  
  - 「表示」と「データ管理」を分けられる。  
  - モバイルアプリや別フロントから同じ API を再利用できる。  
  - バックエンドだけでデータ構造やバリデーションを一元管理できる。

### フロント・バック分離構成

- **良い点**  
  - 役割がはっきりし、チームで分担しやすい。  
  - フロント・バックをそれぞれ差し替えやすく、技術の選択肢が広がる。
- **学習コストが上がる点**  
  - 2 つの環境（Node + Python）と、それぞれの起動手順を覚える必要がある。  
  - CORS や「どの URL にリクエストするか」を意識する必要がある。  
  - エラーが「フロントかバックか」を切り分ける必要がある。

---

## 7. 初心者がつまずきやすいポイント

1. **CORS エラー**  
   ブラウザに「オリジンが違うのでブロック」と出る。→ バックエンドで `CORSMiddleware` を追加し、フロントのオリジン（例: `http://localhost:5173`）を許可する。
2. **API の URL の指定ミス**  
   フロントの fetch 先が `localhost:8000` になっているか、パスが `/tasks` などと一致しているかを確認する。
3. **バックエンドを起動し忘れている**  
   fetch が失敗するときは、先に `uvicorn` でバックエンドが動いているか確認する。
4. **useEffect の依存配列**  
   空 `[]` にすると初回だけ実行。tasks などを依存に含めると、そのたびに API を呼ぶので「無限ループ」に注意する。
5. **状態の更新のタイミング**  
   API が成功してから setState する。失敗時はエラーメッセージを表示するなど、分岐を書く。
6. **データの正規化**  
   タスクの `listId` とリストの `id` が一致しているか。フロントで「現在のリストに属するタスク」でフィルタするか、バックで `GET /lists/{id}/tasks` を用意するか、どちらかで揃える。

---

以上を踏まえ、`backend/` と `frontend/` に実装例を配置する。
