/**
 * タスク入力フォーム（追加）
 * 送信時に親の onAdd を呼ぶ（親が API を呼ぶ）
 */
import React, { useState } from 'react';

function TaskForm({ onAdd }) {
  const [title, setTitle] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const t = title.trim();
    if (!t) return;
    onAdd(t);
    setTitle('');
  };

  return (
    <form className="task-form" onSubmit={handleSubmit}>
      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="タスクを入力"
        maxLength={200}
      />
      <button type="submit">追加</button>
    </form>
  );
}

export default TaskForm;
