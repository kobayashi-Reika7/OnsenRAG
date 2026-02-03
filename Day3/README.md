# Day3 ToDoアプリ（React + FastAPI）

フロントエンドに React（Vite）、バックエンドに FastAPI を使った学習用 ToDo アプリです。

## 全体構成

- **フロントエンド**: React (Vite) — 画面表示・操作・API 呼び出し
- **バックエンド**: FastAPI — タスク・リストの API 提供・データ管理（メモリ）
- **認証**: なし（学習用）

## 環境構築手順

### 1. バックエンド（FastAPI）

1. **Python**  
   Python 3.9 以上をインストールし、`python -V` で確認する。

2. **仮想環境の作成と有効化**
   ```bash
   cd Day3/backend
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **依存関係のインストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **サーバー起動**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   - API: http://localhost:8000  
   - ドキュメント: http://localhost:8000/docs  

### 2. フロントエンド（React）

1. **Node.js**  
   https://nodejs.org/ から LTS をインストールし、`node -v` と `npm -v` で確認する。

2. **依存関係のインストール**
   ```bash
   cd Day3/frontend
   npm install
   ```

3. **開発サーバー起動**
   ```bash
   npm run dev
   ```
   - ブラウザで http://localhost:5173 を開く。

### 3. 動作確認

1. **先にバックエンド**を起動する（`uvicorn`）。
2. 続けて**フロントエンド**を起動する（`npm run dev`）。
3. ブラウザで http://localhost:5173 にアクセスし、タスクの追加・編集・削除ができることを確認する。

## ディレクトリ構成

```
Day3/
├── README.md           # 本ファイル（環境構築・起動手順）
├── docs/
│   └── DESIGN.md       # 設計書（アーキテクチャ・API・考察・つまずきポイント）
├── backend/            # FastAPI
│   ├── main.py         # アプリ・ルート・CORS
│   ├── models.py       # Pydantic モデル
│   ├── store.py        # メモリ上のデータ保持
│   └── requirements.txt
└── frontend/           # React (Vite)
    ├── src/
    │   ├── App.jsx
    │   ├── api.js      # fetch でバックエンド呼び出し
    │   └── components/ # TaskList, TaskItem, TaskForm, Counter, ListSelector, Timer, Memo
    ├── package.json
    └── vite.config.js
```

## API 一覧

| メソッド | パス | 説明 |
|----------|------|------|
| GET | /tasks | タスク一覧取得 |
| POST | /tasks | タスク追加 |
| PUT | /tasks/{id} | タスク更新 |
| DELETE | /tasks/{id} | タスク削除 |
| GET | /lists | リスト一覧取得 |
| POST | /lists | リスト追加 |
| DELETE | /lists/{id} | リスト削除 |

詳細は `docs/DESIGN.md` を参照してください。
