/**
 * 1タスクの表示・編集・削除・お気に入り・期限・タイマー・メモ
 * 状態変更は親の onUpdate を呼び、親が API を呼ぶ
 */
import React, { useState } from 'react';
import Timer from './Timer';
import Memo from './Memo';

function getDueState(dueDate) {
  if (!dueDate) return 'none';
  const today = new Date().toISOString().slice(0, 10);
  if (dueDate < today) return 'overdue';
  return 'ok';
}

function TaskItem({ task, onUpdate, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(task.title);

  const dueState = getDueState(task.due_date);
  const cardClass = [
    'task-card',
    task.is_completed && 'completed',
    dueState === 'overdue' && 'overdue',
    task.is_favorite && 'favorite',
  ]
    .filter(Boolean)
    .join(' ');

  const handleToggleComplete = () => {
    onUpdate(task.id, { is_completed: !task.is_completed });
  };

  const handleToggleFavorite = () => {
    onUpdate(task.id, { is_favorite: !task.is_favorite });
  };

  const handleSaveTitle = () => {
    const t = editTitle.trim();
    if (t) onUpdate(task.id, { title: t });
    setEditing(false);
  };

  const handleDueChange = (e) => {
    const v = e.target.value || null;
    onUpdate(task.id, { due_date: v });
  };

  const handleMemoChange = (memo) => {
    onUpdate(task.id, { memo });
  };

  const handleTimeChange = (time) => {
    onUpdate(task.id, { time });
  };

  return (
    <li className={cardClass}>
      <div className="task-header">
        <input
          type="checkbox"
          checked={task.is_completed}
          onChange={handleToggleComplete}
        />
        {editing ? (
          <>
            <input
              type="text"
              className="task-title"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              onBlur={handleSaveTitle}
              onKeyDown={(e) => e.key === 'Enter' && handleSaveTitle()}
              autoFocus
            />
          </>
        ) : (
          <span className="task-title" onDoubleClick={() => setEditing(true)}>
            {task.title}
          </span>
        )}
        <button
          type="button"
          className={`btn-favorite ${task.is_favorite ? 'active' : ''}`}
          onClick={handleToggleFavorite}
          title="お気に入り"
        >
          {task.is_favorite ? '★' : '☆'}
        </button>
        <button type="button" className="btn-edit" onClick={() => setEditing(!editing)}>
          編集
        </button>
        <button type="button" className="btn-delete" onClick={() => onDelete(task.id)}>
          削除
        </button>
      </div>

      <div className="due-row">
        <input type="date" value={task.due_date || ''} onChange={handleDueChange} />
        <span>{dueState === 'none' ? '期限なし' : dueState === 'ok' ? '期限内' : '期限切れ'}</span>
      </div>

      <Timer
        taskId={task.id}
        initialTime={task.time}
        onTimeChange={handleTimeChange}
      />

      <Memo value={task.memo} onChange={handleMemoChange} />
    </li>
  );
}

export default TaskItem;
