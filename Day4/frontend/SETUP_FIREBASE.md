# React + Firebase 利用の準備手順

## 注意点（重要）

- **`.env` は GitHub に push しない** … API キー等が含まれるため。`.gitignore` で除外済み。
- **`.env.example` は push して OK** … 値が空のテンプレートなので共有してよい。
- **設定後は `npm run dev` を再起動する** … `.env` を変更したら反映のため開発サーバーを再起動すること。

---

## 0. npm が認識されない場合

「npm は、内部または外部コマンドとして認識されていません」と出る場合は、次のいずれかで対処できます。

**A. 起動用スクリプトを使う（手軽）**

- `Day4/frontend` で **`start-dev.bat`** をダブルクリックするか、PowerShell で `.\start-dev.ps1` を実行すると、Node.js をパスに追加してから `npm run dev` が動きます。
- 初回の `npm install` は、下記 B のパス設定後に一度だけ実行してください。

**B. 環境変数 PATH に Node.js を追加する（恒久）**

1. エクスプローラーで **「PC」を右クリック → プロパティ → システムの詳細設定 → 環境変数** を開く。
2. **ユーザー環境変数** の **Path** を選択して **編集** → **新規** で次を追加する。
   ```
   C:\Program Files\nodejs
   ```
3. OK で閉じ、**PowerShell やターミナルを一度閉じてから開き直す**。
4. `node -v` と `npm -v` でバージョンが表示されれば OK。

---

## 1. 依存関係のインストール

Firebase SDK は `package.json` に含まれています。以下でインストールします。

```bash
cd Day4/frontend
npm install
```

（npm がパスにない場合は、上記「0. npm が認識されない場合」の B でパスを追加してから実行してください。）

## 2. Firebase プロジェクトの作成

1. [Firebase Console](https://console.firebase.google.com/) にアクセスし、Google アカウントでログインする。
2. **「プロジェクトを追加」** をクリックし、プロジェクト名を入力して作成する。
3. 作成後、プロジェクトの **歯車アイコン → プロジェクトの設定** を開く。
4. **「一般」** タブで下にスクロールし、**「アプリを追加」** の **</>（Web）** をクリックする。
5. アプリのニックネームを入力（任意）し、**「アプリを登録」** を押す。
6. 表示される **firebaseConfig** の値を控える（後で `.env` に貼り付ける）。

   ```javascript
   apiKey: "xxxx"
   authDomain: "xxxx.firebaseapp.com"
   projectId: "xxxx"
   storageBucket: "xxxx.appspot.com"
   messagingSenderId: "xxxx"
   appId: "xxxx"
   ```

## 3. Firestore データベースの作成

1. Firebase Console の左メニューで **「Firestore Database」** をクリックする。
2. **「データベースを作成」** を押す。
3. **「テストモードで開始」** を選び（学習用）、ロケーションを選んで **「有効にする」** を押す。

## 4. 環境変数ファイルの作成

1. `Day4/frontend` で `.env.example` をコピーし、`.env` という名前で保存する。

   ```bash
   # Windows (PowerShell)
   Copy-Item .env.example .env

   # または手動で .env.example の内容を .env にコピー
   ```

2. `.env` を開き、手順 2 で控えた Firebase の値を入れる。

   ```
   VITE_FIREBASE_API_KEY=ここにapiKeyの値
   VITE_FIREBASE_AUTH_DOMAIN=ここにauthDomainの値
   VITE_FIREBASE_PROJECT_ID=ここにprojectIdの値
   VITE_FIREBASE_STORAGE_BUCKET=ここにstorageBucketの値
   VITE_FIREBASE_MESSAGING_SENDER_ID=ここにmessagingSenderIdの値
   VITE_FIREBASE_APP_ID=ここにappIdの値
   ```

3. **注意**: `.env` は Git にコミットしない（`.gitignore` に含まれています）。API キーがリポジトリに載らないようにするためです。
4. **`.env` を変更したら `npm run dev` を再起動する**（反映のため）。

## 5. 起動と確認

```bash
npm run dev
```

ブラウザで http://localhost:5173 を開き、タスクを追加してリロードしても消えないことを確認する。Firebase Console の Firestore で `todos` コレクションにドキュメントが増えていれば準備完了です。
