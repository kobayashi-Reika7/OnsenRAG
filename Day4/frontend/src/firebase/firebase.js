/**
 * Firebase 初期化のみを行うファイル
 * ここではCRUD処理を書かず、Firestoreインスタンス(db)をexportする責務だけを持つ
 */
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

// Viteでは .env の変数は import.meta.env.VITE_* で参照する（クライアントに公開される）
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

// アプリを1回だけ初期化（複数回呼ぶとエラーになるため、このファイルで一元管理）
const app = initializeApp(firebaseConfig);

// Firestore インスタンスを取得し、他ファイルから import して使う
export const db = getFirestore(app);
