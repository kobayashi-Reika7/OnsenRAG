# Day4 クラウドデプロイガイド

Day4 をクラウド上で動かすための設定と手順です。

## 1. 構成の整理

| 構成 | 説明 |
|------|------|
| **フロントエンド** | React (Vite) + Firebase Firestore。Vercel / Netlify などにデプロイ可能。 |
| **バックエンド（任意）** | FastAPI。Day4 のメインは Firestore のみだが、参考用の API を Railway / Render / Cloud Run などにデプロイ可能。 |

- **Firestore のみで使う場合**: フロントだけデプロイすればよい。
- **バックエンド API も使う場合**: フロントの `VITE_API_BASE` にバックエンドの URL を設定する。

---

## 2. フロントエンドのクラウドデプロイ

### 2.1 環境変数（ビルド時に必要）

デプロイ先の「環境変数」に以下を設定する。

| 変数 | 説明 | 必須 |
|------|------|------|
| `VITE_FIREBASE_API_KEY` | Firebase API キー | ○ |
| `VITE_FIREBASE_AUTH_DOMAIN` | Firebase Auth ドメイン | ○ |
| `VITE_FIREBASE_PROJECT_ID` | Firebase プロジェクト ID | ○ |
| `VITE_FIREBASE_STORAGE_BUCKET` | Storage バケット | ○ |
| `VITE_FIREBASE_MESSAGING_SENDER_ID` | メッセージング Sender ID | ○ |
| `VITE_FIREBASE_APP_ID` | Firebase アプリ ID | ○ |
| `VITE_API_BASE` | バックエンド API のベース URL（API を使う場合のみ） | 任意 |

### 2.2 Vercel

1. リポジトリを Vercel に連携。
2. **Root Directory** を `Day4/frontend` に設定。
3. **Build Command**: `npm run build`
4. **Output Directory**: `dist`
5. 上記の環境変数を「Environment Variables」に追加（VITE_ 付き）。
6. デプロイ後、Firebase Console の「認証」→「認証ドメイン」に、Vercel のドメイン（例: `xxx.vercel.app`）を追加する。

### 2.3 Netlify

1. リポジトリを Netlify に連携。
2. **Base directory**: `Day4/frontend`
3. **Build command**: `npm run build`
4. **Publish directory**: `Day4/frontend/dist`
5. 環境変数を「Site settings」→「Environment variables」に追加。
6. Firebase の認証ドメインに Netlify のドメインを追加。

---

## 3. バックエンド（FastAPI）のクラウドデプロイ

### 3.1 環境変数

| 変数 | 説明 | デフォルト |
|------|------|------------|
| `ALLOWED_ORIGINS` | CORS 許可オリジン（カンマ区切り） | `http://localhost:5173,http://127.0.0.1:5173` |
| `PORT` | 待ち受けポート（多くの PaaS が自動設定） | `8000` |

本番では `ALLOWED_ORIGINS` にフロントの URL を指定する（例: `https://myapp.vercel.app`）。

### 3.2 Docker で起動（任意）

```bash
cd Day4/backend
docker build -t day4-todo-api .
docker run -p 8000:8000 \
  -e ALLOWED_ORIGINS=https://your-frontend.vercel.app \
  day4-todo-api
```

### 3.3 Railway

1. リポジトリまたは Docker でデプロイ。
2. **Root Directory**: `Day4/backend`
3. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`（未指定時は Railway が PORT を注入）。
4. 環境変数に `ALLOWED_ORIGINS` を設定（フロントの URL）。
5. 発行された URL（例: `https://xxx.railway.app`）をフロントの `VITE_API_BASE` に設定して再ビルド。

### 3.4 Render

1. **New** → **Web Service**。
2. リポジトリを指定。**Root Directory**: `Day4/backend`。
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. 環境変数に `ALLOWED_ORIGINS` を追加。
6. 発行された URL をフロントの `VITE_API_BASE` に設定。

### 3.5 Google Cloud Run（Docker 利用）

1. イメージをビルド・プッシュ:
   ```bash
   cd Day4/backend
   gcloud builds submit --tag gcr.io/PROJECT_ID/day4-todo-api
   ```
2. サービス作成時に `ALLOWED_ORIGINS` を環境変数で渡す。
3. サービス URL をフロントの `VITE_API_BASE` に設定。

### 3.6 ヘルスチェック

バックエンドは `/health` を用意している。ロードバランサやオーケストレーションのヘルスチェック先に指定できる。

```http
GET /health
→ {"status":"ok"}
```

---

## 4. 本番時の注意

- **Firebase**: 本番ドメインを Firebase Console の認証ドメインに登録する。
- **CORS**: バックエンドを使う場合は、`ALLOWED_ORIGINS` にフロントの **本番 URL のみ** を指定し、必要なら複数カンマ区切りで追加する。
- **API URL**: フロントの `VITE_API_BASE` は **ビルド時** に埋め込まれるため、変更したら再ビルド・再デプロイが必要。
- **シークレット**: `.env` はリポジトリに含めず、各クラウドの「環境変数」で設定する。

---

## 5. ローカルでクラウドと同じ動きを試す

- フロント: `.env` に Firebase 設定 +（必要なら）`VITE_API_BASE=http://localhost:8000`
- バックエンド: `.env` に `ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`（未設定でもデフォルトで同じ）
- バックエンド起動: `uvicorn main:app --host 0.0.0.0 --port 8000` または `PORT=8000 uvicorn main:app --host 0.0.0.0 --port 8000`
