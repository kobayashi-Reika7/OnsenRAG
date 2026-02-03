/**
 * Firebase 接続設定
 *
 * - initializeApp: Firebase アプリを初期化
 * - getFirestore: Firestore インスタンスを取得（クラウドDB への接続）
 * - 設定値は import.meta.env から取得
 *
 * この db を使って、ToDo を Firestore に保存・取得する。
 * データは Google のサーバーに永続化され、リロード・タブを閉じても消えない。
 */
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

// .env から Firebase の設定値を取得
// 直接値を書かず、必ず import.meta.env を使う（セキュリティ・環境ごとの切り替えのため）
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

// Firebase アプリを初期化（複数回呼ぶとエラーになるため、このファイルで1回だけ）
const app = initializeApp(firebaseConfig);

// Firestore インスタンスを取得し、他ファイルから import して使う
export const db = getFirestore(app);
