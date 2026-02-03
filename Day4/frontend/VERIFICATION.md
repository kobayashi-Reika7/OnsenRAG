# Day4 動作確認チェックリスト

## リロード後も残らない場合の確認

1. **Firestore ルール**: [Firebase Console](https://console.firebase.google.com/) → Firestore → ルール で、`lists` / `todos` コレクションの読み取りが許可されているか確認
2. **エラー表示**: 画面上にエラーが出る場合は、ルールまたはネットワークを確認

## 起動

```bash
cd Day4/frontend
npm install
npm run dev
```

- **URL**: http://localhost:5100

---

## 手動検証手順

### 1. 初回表示
- [ ] 「ToDoアプリ」が見出しで表示される
- [ ] カウンター・リスト選択・タスク入力欄が表示される
- [ ] リストに「マイリスト」が 1 件ある

### 2. タスク追加・反映
- [ ] タスクを入力して「追加」→ 一覧に表示される
- [ ] リロード後もタスクが残る

### 3. リスト・永続化
- [ ] リスト追加・切替・削除ができる
- [ ] タブを閉じて開き直してもデータが残る
- [ ] Firebase Console で `lists` / `todos` にデータがある
