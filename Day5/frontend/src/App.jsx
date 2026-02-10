/**
 * ルーティング管理と認証状態
 * 未ログイン時は /login へリダイレクト。認証状態は Context で共有
 */
import React, { useState, useEffect, createContext, useContext, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { subscribeAuth } from './services/auth';

// 遅延ロード: 初期バンドルサイズを削減
const TopPage = React.lazy(() => import('./pages/TopPage'));
const LoginPage = React.lazy(() => import('./pages/LoginPage'));
const SignupPage = React.lazy(() => import('./pages/SignupPage'));
const MenuPage = React.lazy(() => import('./pages/MenuPage'));
const CalendarPage = React.lazy(() => import('./pages/CalendarPage'));
const ReservationFormPage = React.lazy(() => import('./pages/ReservationFormPage'));
const ReserveConfirmPage = React.lazy(() => import('./pages/ReserveConfirmPage'));
const MyReservationsPage = React.lazy(() => import('./pages/MyReservationsPage'));

const AuthContext = createContext(null);

export function useAuth() {
  const ctx = useContext(AuthContext);
  return ctx ?? null;
}

/**
 * Error Boundary: 予期しない JS エラーでアプリ全体がクラッシュするのを防ぐ
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    if (import.meta.env.DEV) {
      console.error('ErrorBoundary caught:', error, errorInfo);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="page" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
          <h1 style={{ fontSize: '1.3rem', marginBottom: '1rem' }}>エラーが発生しました</h1>
          <p style={{ marginBottom: '1.5rem' }}>予期しないエラーが発生しました。ページを再読み込みしてください。</p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => window.location.reload()}
          >
            ページを再読み込み
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

/**
 * ログイン必須のルート。未ログインなら /login へリダイレクト
 */
function ProtectedRoute({ children }) {
  const user = useAuth();
  const location = useLocation();
  if (user === undefined) {
    return <div className="app-loading">読み込み中…</div>;
  }
  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return children;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route
        path="/menu"
        element={
          <ProtectedRoute>
            <MenuPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/calendar"
        element={
          <ProtectedRoute>
            <CalendarPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/reserve/form"
        element={
          <ProtectedRoute>
            <ReservationFormPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/reserve/confirm"
        element={
          <ProtectedRoute>
            <ReserveConfirmPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/reservations"
        element={
          <ProtectedRoute>
            <MyReservationsPage />
          </ProtectedRoute>
        }
      />
      <Route path="/" element={<TopPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  const [user, setUser] = useState(undefined);

  useEffect(() => {
    const unsubscribe = subscribeAuth((u) => setUser(u ?? null));
    return () => unsubscribe();
  }, []);

  return (
    <ErrorBoundary>
      <AuthContext.Provider value={user}>
        <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <div className="app">
            <Suspense fallback={<div className="app-loading">読み込み中…</div>}>
              <AppRoutes />
            </Suspense>
          </div>
        </BrowserRouter>
      </AuthContext.Provider>
    </ErrorBoundary>
  );
}

export default App;
