/**
 * Firestore の保存・取得（学習用シンプル仕様）
 * - lists: リスト一覧（デフォルト「マイリスト」を1件持つ）
 * - todos: タスク（title, list_id 必須。完了・お気に入り・期限・メモ・タイマーは任意）
 *
 * 【リロード後も残る】onSnapshot でリアルタイム監視。orderBy は使わずクライアントでソート（インデックス不要）。
 */
import {
  collection,
  addDoc,
  getDocs,
  onSnapshot,
  deleteDoc,
  doc,
  updateDoc,
  serverTimestamp,
  query,
  where,
  writeBatch,
} from 'firebase/firestore';
import { db } from '../firebase/firebase';

const LISTS = 'lists';
const TASKS = 'todos';

// ========== リスト ==========

const DEFAULT_LIST_NAME = 'マイリスト';

export function getLists() {
  const ref = collection(db, LISTS);
  return getDocs(ref).then((snap) => {
    const items = snap.docs.map((d) => ({
      id: d.id,
      name: (d.data().name ?? '').trim() || '（無題）',
    }));
    return [...items].sort((a, b) => {
      const aDefault = a.name === DEFAULT_LIST_NAME ? 0 : 1;
      const bDefault = b.name === DEFAULT_LIST_NAME ? 0 : 1;
      return aDefault - bDefault;
    });
  });
}

/**
 * リストをリアルタイム監視（onSnapshot）
 * リロード後も確実にデータを取得。orderBy 未使用でインデックス不要。
 * @param {function(lists: Array<{id, name}>): void} callback
 * @param {function(error: Error): void} [onError]
 * @returns {function} unsubscribe
 */
export function subscribeLists(callback, onError) {
  const ref = collection(db, LISTS);
  return onSnapshot(
    ref,
    (snap) => {
      const items = snap.docs.map((d) => ({
        id: d.id,
        name: (d.data().name ?? '').trim() || '（無題）',
      }));
      const sorted = [...items].sort((a, b) => {
        const aDefault = a.name === DEFAULT_LIST_NAME ? 0 : 1;
        const bDefault = b.name === DEFAULT_LIST_NAME ? 0 : 1;
        return aDefault - bDefault;
      });
      callback(sorted);
    },
    (err) => {
      console.error('Firestore lists 監視エラー:', err);
      if (onError) onError(err);
      callback([]);
    }
  );
}

export function addList(name) {
  const ref = collection(db, LISTS);
  const payload = { name: (name ?? '').trim() || '（無題）' };
  return addDoc(ref, payload).then((docRef) => ({
    id: docRef.id,
    name: payload.name,
  }));
}

export function deleteList(listId, defaultListId) {
  const taskRef = collection(db, TASKS);
  const q = query(taskRef, where('list_id', '==', listId));
  return getDocs(q)
    .then((snap) => {
      const batch = writeBatch(db);
      snap.docs.forEach((d) => {
        batch.update(doc(db, TASKS, d.id), { list_id: defaultListId });
      });
      return batch.commit();
    })
    .then(() => deleteDoc(doc(db, LISTS, listId)));
}

// ========== タスク（保存・取得を確実に） ==========

/**
 * 全タスクを取得。list_id がないドキュメントは defaultListId で補う
 */
export function getTasks(defaultListId) {
  const ref = collection(db, TASKS);
  const fallback = defaultListId ?? '';
  return getDocs(ref).then((snap) =>
    snap.docs.map((d) => mapDocToTask(d, fallback))
  );
}

function mapDocToTask(d, fallbackListId) {
  const data = d.data();
  return {
    id: d.id,
    title: data.title ?? '',
    list_id: data.list_id ?? fallbackListId,
    is_completed: Boolean(data.is_completed),
    is_favorite: Boolean(data.is_favorite),
    due_date: data.due_date ?? null,
    memo: data.memo ?? '',
    time: Number(data.time) || 0,
    createdAt: data.createdAt ?? null,
  };
}

/**
 * タスクをリアルタイム監視（onSnapshot）
 * リロード後も確実にデータを取得。orderBy 未使用でインデックス不要。
 * @param {string} defaultListId - list_id 欠損時のフォールバック
 * @param {function(tasks: Array): void} callback
 * @param {function(error: Error): void} [onError]
 * @returns {function} unsubscribe
 */
export function subscribeTasks(defaultListId, callback, onError) {
  const ref = collection(db, TASKS);
  const fallback = defaultListId ?? '';
  return onSnapshot(
    ref,
    (snap) => {
      const tasks = snap.docs.map((d) => mapDocToTask(d, fallback));
      callback(tasks);
    },
    (err) => {
      console.error('Firestore tasks 監視エラー:', err);
      if (onError) onError(err);
      callback([]);
    }
  );
}

/**
 * タスクを1件追加（必須: title, list_id）
 */
export function addTask({ title, list_id, is_completed = false, is_favorite = false, due_date = null, memo = '', time = 0 }) {
  const ref = collection(db, TASKS);
  return addDoc(ref, {
    title: String(title ?? '').trim(),
    list_id: String(list_id ?? ''),
    is_completed: Boolean(is_completed),
    is_favorite: Boolean(is_favorite),
    due_date: due_date ?? null,
    memo: String(memo ?? ''),
    time: Number(time) || 0,
    createdAt: serverTimestamp(),
  }).then((docRef) => docRef.id);
}

/**
 * タスクを更新（渡したフィールドだけ上書き）
 */
export function updateTask(id, data) {
  const ref = doc(db, TASKS, id);
  const payload = { ...data };
  if (payload.title !== undefined) payload.title = String(payload.title).trim();
  return updateDoc(ref, payload);
}

/**
 * タスクを1件削除
 */
export function deleteTask(id) {
  return deleteDoc(doc(db, TASKS, id));
}
