/**
 * App: 画面の状態（state）と Firestore の橋渡し
 *
 * 【state の役割】
 * - 画面上の「今表示しているタスク一覧」を保持する
 * - ユーザー操作に応じてすぐ UI を更新するため（ローカルで一時的に持つ）
 *
 * 【Firestore の役割】
 * - データの永続化（リロードしても消えない）
 * - 追加・取得・削除はすべて Firestore 経由で行い、完了後に state を更新する
 *
 * 流れ: 初回読み込み → getTodos() で Firestore から取得 → setTasks で state に反映
 *       追加 → addTodo() で Firestore に保存 → getTodos() で再取得して state 更新
 *       削除 → deleteTodo() で Firestore から削除 → getTodos() で再取得して state 更新
 *
 * 【表示ズレの原因（Firestore 対応後に起きていたこと）】
 * 1. 並び順: getTodos() は orderBy なしで取得するため、Firestore の返却順は保証されない。
 *    ローカル版は「新規が先頭」だったが、クラウドでは並びが毎回変わりうる。
 * 2. 表示タイミング: 追加後に setInputValue('') を先に実行していたため、
 *    一覧の再取得（loadTodos）が終わる前に入力欄だけ空になり、一瞬「リストと入力がズレた」状態になっていた。
 */
import React, { useState, useEffect } from 'react';
import { addTodo, getTodos, deleteTodo } from './services/firestore';
import TodoList from './components/TodoList';

function App() {
  const [tasks, setTasks] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /**
   * Firestore から一覧を取得し、state を更新する。
   * 並び順を安定させるため、取得後に createdAt の新しい順でソートしてから setTasks する。
   */
  const loadTodos = async () => {
    try {
      setError(null);
      const list = await getTodos();
      // 並び順を安定させる: createdAt の新しい順（ローカル版と同じ「新規が上」）
      const sorted = [...list].sort((a, b) => {
        const tA = a.createdAt?.toMillis ? a.createdAt.toMillis() : (a.createdAt ?? 0);
        const tB = b.createdAt?.toMillis ? b.createdAt.toMillis() : (b.createdAt ?? 0);
        return tB - tA;
      });
      setTasks(sorted);
    } catch (e) {
      setError('タスクの取得に失敗しました: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  // 初回表示時に1回だけ getTodos を呼ぶ（依存配列が空なのでマウント時のみ）
  useEffect(() => {
    loadTodos();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const title = inputValue.trim();
    if (!title) return;
    try {
      setError(null);
      await addTodo(title);
      await loadTodos();
      // 一覧の更新が完了してから入力欄を空にする（表示タイミングのズレを防ぐ）
      setInputValue('');
    } catch (e) {
      setError('タスクの追加に失敗しました: ' + e.message);
    }
  };

  const handleDelete = async (id) => {
    try {
      setError(null);
      await deleteTodo(id);
      await loadTodos();
    } catch (e) {
      setError('タスクの削除に失敗しました: ' + e.message);
    }
  };

  return (
    <div style={{ maxWidth: 400, margin: '2rem auto', padding: '0 1rem' }}>
      <h1>ToDoアプリ (React + Firestore)</h1>
      {error && <p style={{ color: '#c33' }}>{error}</p>}

      <form onSubmit={handleSubmit} style={{ marginBottom: '1rem' }}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="新しいタスク"
          aria-label="新しいタスクの入力"
        />
        <button type="submit">追加</button>
      </form>

      {loading ? (
        <p>読み込み中...</p>
      ) : (
        <TodoList tasks={tasks} onDelete={handleDelete} />
      )}
    </div>
  );
}

export default App;
