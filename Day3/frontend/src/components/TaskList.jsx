/**
 * タスク一覧表示（現在リストに属するタスクのみ）
 * 各 TaskItem に onUpdate / onDelete を渡す
 */
import React from 'react';
import TaskItem from './TaskItem';

function TaskList({ tasks, currentListId, onUpdate, onDelete }) {
  const filtered = tasks.filter((t) => t.list_id === currentListId);

  return (
    <ul className="task-list">
      {filtered.map((task) => (
        <TaskItem
          key={task.id}
          task={task}
          onUpdate={onUpdate}
          onDelete={onDelete}
        />
      ))}
    </ul>
  );
}

export default TaskList;
