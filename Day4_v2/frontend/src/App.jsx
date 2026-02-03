/**
 * クラウド対応 ToDo アプリ
 *
 * 【動作保証】
 * - ブラウザをリロード → ToDo が残る
 * - タブを閉じて開き直す → ToDo が残る
 * - 別タブで追加 → 即反映される
 *
 * 【なぜ ToDo が消えないのか】
 * useState だけだと、リロードやタブを閉じるとメモリが解放されデータは消える。
 * 本アプリは Firestore（クラウドDB）に保存しているため、
 * データは Google のサーバーに永続化され、いつでも取得できる。
 *
 * 【useState と Firestore の役割の違い】
 * - useState: 「今この画面に表示する値」を一時的に持つ（メモリ上）
 * - Firestore: 「正のデータ」をクラウドに永続保存
 * フロー: ページ表示 → Firestore から取得 → setTodos で useState に反映 → 画面描画
 */
import { useState, useEffect } from 'react';
import { subscribeTodos, addTodoToDB } from './services/firestore.js';
import './App.css';

function App() {
  // useState: 画面に表示する ToDo 一覧（Firestore から取得した値をここに格納）
  const [todos, setTodos] = useState([]);

  // 入力欄の値（フォームの制御用）
  const [inputTitle, setInputTitle] = useState('');

  const [error, setError] = useState(null);

  /**
   * 【初回表示 + リアルタイム更新】subscribeTodos（onSnapshot）
   *
   * onSnapshot の仕組み:
   * 1. ページ読み込み時 → Firestore に接続し、リスナーを登録
   * 2. 初回 → 即座に現在のデータを callback で渡す → setTodos で表示
   * 3. 誰かが追加・変更・削除すると → Firestore が変更を検知
   * 4. 検知後 → 自動で callback が呼ばれる → setTodos で画面更新
   * 5. 同じページを別タブで開いていても、そのタブでも callback が呼ばれる
   *
   * これにより「別タブで追加 → 即反映」が実現する。
   */
  useEffect(() => {
    const unsubscribe = subscribeTodos((data) => {
      setTodos(data);
    });

    // cleanup: タブを閉じる等でコンポーネントが unmount されたら監視を解除
    return () => {
      unsubscribe();
    };
  }, []);

  /**
   * フォーム送信: Firestore に追加
   * 追加後、onSnapshot が自動で検知して setTodos が呼ばれるため、手動再取得不要
   */
  async function handleSubmit(e) {
    e.preventDefault();

    const title = inputTitle.trim();
    if (!title) return;

    try {
      setError(null);
      await addTodoToDB(title);
      setInputTitle('');
      // subscribeTodos が自動で検知するため、fetchTodos のような再取得は不要
    } catch (err) {
      console.error('ToDo 追加エラー:', err);
      setError('追加に失敗しました');
    }
  }

  return (
    <div className="app">
      <h1>ToDoアプリ</h1>

      {/* 追加フォーム */}
      <form onSubmit={handleSubmit} className="todo-form">
        <input
          type="text"
          value={inputTitle}
          onChange={(e) => setInputTitle(e.target.value)}
          placeholder="タスクを入力"
          maxLength={200}
        />
        <button type="submit">追加</button>
      </form>

      {/* エラー表示 */}
      {error && <p className="error">{error}</p>}

      {/* ToDo 一覧（新しい順で表示・リアルタイム更新） */}
      <ul className="todo-list">
        {todos.map((todo) => (
          <li key={todo.id}>{todo.title}</li>
        ))}
      </ul>

      {todos.length === 0 && (
        <p className="empty">ToDo がありません。追加してください。</p>
      )}
    </div>
  );
}

export default App;
