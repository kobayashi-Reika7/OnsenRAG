/**
 * TodoList: props で受け取ったタスク配列を表示するだけのコンポーネント
 * データの取得・追加・削除は親（App）が担当し、ここでは「一覧表示」の責務のみ
 */
import React from 'react';

function TodoList({ tasks, onDelete }) {
  return (
    <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
      {tasks.map((task) => (
        <li
          key={task.id}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            marginBottom: '0.5rem',
          }}
        >
          <span style={{ flex: 1 }}>{task.title || '(無題)'}</span>
          <button
            type="button"
            onClick={() => onDelete(task.id)}
            aria-label={`「${task.title}」を削除`}
          >
            削除
          </button>
        </li>
      ))}
    </ul>
  );
}

export default TodoList;
