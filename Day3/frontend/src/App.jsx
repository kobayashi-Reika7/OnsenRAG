/**
 * App: 全体の状態保持と API 連携
 * tasks, lists, currentListId を保持し、fetch でバックエンドと通信する
 */
import React, { useState, useEffect, useCallback } from 'react';
import { getTasks, createTask, updateTask, deleteTask, getLists, createList, deleteList } from './api';
import Counter from './components/Counter';
import ListSelector from './components/ListSelector';
import TaskForm from './components/TaskForm';
import TaskList from './components/TaskList';

function getDueState(dueDate) {
  if (!dueDate) return 'none';
  const today = new Date().toISOString().slice(0, 10);
  if (dueDate < today) return 'overdue';
  return 'ok';
}

function computeCounts(tasks) {
  const incomplete = tasks.filter((t) => !t.is_completed).length;
  const completed = tasks.filter((t) => t.is_completed).length;
  const favorite = tasks.filter((t) => t.is_favorite).length;
  const overdue = tasks.filter((t) => getDueState(t.due_date) === 'overdue').length;
  return { incomplete, completed, favorite, overdue };
}

function App() {
  const [tasks, setTasks] = useState([]);
  const [lists, setLists] = useState([]);
  const [currentListId, setCurrentListId] = useState(1);
  const [error, setError] = useState(null);

  const loadTasks = useCallback(async () => {
    try {
      setError(null);
      const data = await getTasks();
      setTasks(data);
    } catch (e) {
      setError('タスクの取得に失敗しました: ' + e.message);
    }
  }, []);

  const loadLists = useCallback(async () => {
    try {
      setError(null);
      const data = await getLists();
      setLists(data);
    } catch (e) {
      setError('リストの取得に失敗しました: ' + e.message);
    }
  }, []);

  useEffect(() => {
    loadTasks();
    loadLists();
  }, [loadTasks, loadLists]);

  const handleAddTask = async (title) => {
    try {
      setError(null);
      await createTask({ title, list_id: currentListId });
      await loadTasks();
    } catch (e) {
      setError('タスクの追加に失敗しました: ' + e.message);
    }
  };

  const handleUpdateTask = async (id, data) => {
    try {
      setError(null);
      await updateTask(id, data);
      await loadTasks();
    } catch (e) {
      setError('タスクの更新に失敗しました: ' + e.message);
    }
  };

  const handleDeleteTask = async (id) => {
    try {
      setError(null);
      await deleteTask(id);
      await loadTasks();
    } catch (e) {
      setError('タスクの削除に失敗しました: ' + e.message);
    }
  };

  const handleAddList = async () => {
    const name = window.prompt('新しいリスト名を入力してください');
    if (!name?.trim()) return;
    try {
      setError(null);
      const created = await createList(name.trim());
      await loadLists();
      setCurrentListId(created.id);
    } catch (e) {
      setError('リストの追加に失敗しました: ' + e.message);
    }
  };

  const handleDeleteList = async () => {
    if (currentListId === 1) {
      window.alert('デフォルトリストは削除できません');
      return;
    }
    if (!window.confirm('このリストを削除しますか？\n属するタスクはデフォルトリストに移動します。')) return;
    try {
      setError(null);
      await deleteList(currentListId);
      await loadLists();
      await loadTasks();
      setCurrentListId(1);
    } catch (e) {
      setError('リストの削除に失敗しました: ' + e.message);
    }
  };

  const counts = computeCounts(tasks);

  return (
    <>
      <h1>ToDoアプリ (React + FastAPI)</h1>
      {error && <p style={{ color: '#c33' }}>{error}</p>}
      <Counter counts={counts} />
      <ListSelector
        lists={lists}
        currentListId={currentListId}
        onSelect={setCurrentListId}
        onAdd={handleAddList}
        onDelete={handleDeleteList}
      />
      <TaskForm onAdd={handleAddTask} />
      <TaskList
        tasks={tasks}
        currentListId={currentListId}
        onUpdate={handleUpdateTask}
        onDelete={handleDeleteTask}
      />
    </>
  );
}

export default App;
