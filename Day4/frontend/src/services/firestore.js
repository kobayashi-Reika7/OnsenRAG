/**
 * Firestore を使った ToDo の CRUD 処理
 * todos コレクションに対して追加・取得・削除を行う（async/await で統一）
 */
import {
  collection,
  addDoc,
  getDocs,
  deleteDoc,
  doc,
  serverTimestamp,
} from 'firebase/firestore';
import { db } from '../firebase/firebase';

const COLLECTION_NAME = 'todos';

/**
 * タスクを1件追加する
 * @param {string} title - タスクのタイトル
 * @returns {Promise<string>} 作成されたドキュメントのID
 */
export async function addTodo(title) {
  const colRef = collection(db, COLLECTION_NAME);
  const docRef = await addDoc(colRef, {
    title: title.trim(),
    createdAt: serverTimestamp(), // サーバー時刻で統一（クライアントのずれを防ぐ）
  });
  return docRef.id;
}

/**
 * 全タスクを取得する
 * （orderBy はインデックスが必要なため未使用。必要なら取得後にJSでソート可能）
 * @returns {Promise<Array<{id: string, title: string, createdAt: object}>>}
 */
export async function getTodos() {
  const colRef = collection(db, COLLECTION_NAME);
  const snapshot = await getDocs(colRef);
  return snapshot.docs.map((d) => ({
    id: d.id,
    title: d.data().title ?? '',
    createdAt: d.data().createdAt ?? null,
  }));
}

/**
 * 指定IDのタスクを削除する
 * @param {string} id - ドキュメントID
 */
export async function deleteTodo(id) {
  const docRef = doc(db, COLLECTION_NAME, id);
  await deleteDoc(docRef);
}
