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
 */
import React, { useState, useEffect } from 'react';
import { addTodo, getTodos, deleteTodo } from './services/firestore';
import TodoList from './components/TodoList';

function App() {
  const [tasks, setTasks] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // 初回マウント時と、追加・削除後に Firestore から一覧を取得する
  const loadTodos = async () => {
    try {
      setError(null);
      const list = await getTodos();
      setTasks(list);
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
      setInputValue('');
      await loadTodos();
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
