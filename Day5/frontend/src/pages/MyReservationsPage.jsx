/**
 * 予約一覧画面（予約確認）
 * users/{uid}/reservations を取得し、カード形式で表示。変更・キャンセル操作を安全に提供。
 */
import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../App';
import Breadcrumb from '../components/Breadcrumb';
import { getReservationsByUser } from '../services/reservation';
import { cancelReservationApi, invalidateSlotsCache } from '../services/backend';
import { invalidateAvailabilityCache } from '../services/availability';
import { logout } from '../services/auth';
import { getAuth } from 'firebase/auth';

/** date: YYYY-MM-DD, time: HH:mm → 表示用 "2026/02/10 10:30" */
function formatDateTime(dateStr, timeStr) {
  if (!dateStr) return '—';
  const parts = dateStr.split('-').filter(Boolean);
  if (parts.length < 3) return dateStr;
  const dateFormatted = parts.join('/');
  const time = (timeStr || '').trim();
  return time ? `${dateFormatted} ${time}` : dateFormatted;
}

/** 予約が過去かどうか判定（日付+時間で比較） */
function isPastReservation(dateStr, timeStr) {
  if (!dateStr) return false;
  const now = new Date();
  const todayYmd = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  const nowTime = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
  if (dateStr < todayYmd) return true;
  if (dateStr === todayYmd && timeStr && timeStr <= nowTime) return true;
  return false;
}

/** キャンセル確認ダイアログ（Escape/バックドロップクリック/フォーカス管理） */
function CancelDialog({ cancelTarget, cancellingId, onConfirm, onClose }) {
  const dialogRef = useRef(null);
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && !cancellingId) onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    // フォーカスをダイアログ内に移動
    const firstBtn = dialogRef.current?.querySelector('button');
    if (firstBtn) firstBtn.focus();
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [cancellingId, onClose]);
  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget && !cancellingId) onClose();
  };
  return (
    <div className="reservation-cancel-overlay" role="dialog" aria-modal="true" aria-labelledby="cancel-dialog-title" onClick={handleOverlayClick}>
      <div className="reservation-cancel-dialog" ref={dialogRef}>
        <h2 id="cancel-dialog-title" className="reservation-cancel-dialog-title">キャンセルの確認</h2>
        <p className="reservation-cancel-dialog-message">
          この予約をキャンセルしますか？一度キャンセルすると元に戻せません。
        </p>
        <p className="reservation-cancel-dialog-summary">{cancelTarget.summary}</p>
        <div className="reservation-cancel-dialog-actions">
          <button type="button" className="btn reservation-btn-cancel-confirm" onClick={onConfirm} disabled={cancellingId === cancelTarget.id}>
            {cancellingId === cancelTarget.id ? '処理中…' : 'キャンセルする'}
          </button>
          <button type="button" className="btn btn-secondary" onClick={onClose} disabled={!!cancellingId}>
            戻る
          </button>
        </div>
      </div>
    </div>
  );
}

function MyReservationsPage() {
  const navigate = useNavigate();
  const user = useAuth();
  const [reservations, setReservations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [cancelTarget, setCancelTarget] = useState(null);
  const [cancellingId, setCancellingId] = useState(null);

  const fetchReservations = useCallback(() => {
    if (!user?.uid) return;
    setLoading(true);
    setError('');
    getReservationsByUser(user.uid)
      .then(setReservations)
      .catch((err) => setError(err?.message ?? '予約一覧の取得に失敗しました。'))
      .finally(() => setLoading(false));
  }, [user?.uid]);

  useEffect(() => {
    fetchReservations();
  }, [fetchReservations]);

  const handleCancelClick = (r) => {
    setCancelTarget({ id: r.id, summary: formatDateTime(r.date, r.time) + ' ' + (r.department || '') });
  };

  const handleCancelConfirm = async () => {
    if (!user?.uid || !cancelTarget) return;
    setCancellingId(cancelTarget.id);
    setError('');
    try {
      // バックエンド API 経由でキャンセル（booked_slots も同時に解放される）
      const auth = getAuth();
      const idToken = await auth.currentUser?.getIdToken();
      if (!idToken) throw new Error('認証情報がありません。再ログインしてください。');
      await cancelReservationApi(idToken, cancelTarget.id);
      invalidateSlotsCache();
      invalidateAvailabilityCache();
      setCancelTarget(null);
      setSuccessMessage('予約をキャンセルしました。');
      fetchReservations();
      setTimeout(() => setSuccessMessage(''), 4000);
    } catch (err) {
      setError(err?.message ?? 'キャンセルに失敗しました。');
    } finally {
      setCancellingId(null);
    }
  };

  const handleChangeClick = (r) => {
    navigate('/reserve/form', { state: { isEditing: true, editingReservationId: r.id, editingReservation: r } });
  };

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login', { replace: true });
    } catch {
      navigate('/login', { replace: true });
    }
  };

  // 今後の予約（日付の近い順）と過去の予約（日付の新しい順）に分離
  const { upcoming, past } = useMemo(() => {
    const up = [];
    const pa = [];
    for (const r of reservations) {
      if (isPastReservation(r.date, r.time)) {
        pa.push(r);
      } else {
        up.push(r);
      }
    }
    up.sort((a, b) => (a.date + a.time).localeCompare(b.date + b.time));
    pa.sort((a, b) => (b.date + b.time).localeCompare(a.date + a.time));
    return { upcoming: up, past: pa };
  }, [reservations]);

  const [showHistory, setShowHistory] = useState(false);

  return (
    <div className="page page-reservations">
      <Breadcrumb
        items={[
          { label: 'Top', to: '/' },
          { label: 'メニュー', to: '/menu' },
          { label: '予約一覧' },
        ]}
      />
      <header className="reservations-header">
        <h1 className="reservations-title">予約一覧</h1>
      </header>

      {loading && (
        <div className="reservations-loading" aria-live="polite">
          <span className="reservations-loading-spinner" aria-hidden />
          読み込み中…
        </div>
      )}
      {error && <p className="page-error" role="alert">{error}</p>}
      {successMessage && <p className="reservations-success" role="status">{successMessage}</p>}

      {/* 今後の予約 */}
      {!loading && !error && upcoming.length > 0 && (
        <>
          <h2 className="reservations-section-title">今後の予約</h2>
          <ul className="reservations-list" aria-label="今後の予約">
            {upcoming.map((r) => (
              <li key={r.id} className="reservation-card">
                <p className="reservation-card-datetime">
                  <span className="reservation-card-datetime-icon" aria-hidden>📅</span>
                  {formatDateTime(r.date, r.time)}
                </p>
                <p className="reservation-card-meta">
                  {[r.department || '—', r.purpose || '—'].filter(Boolean).join(' / ')}
                </p>
                <div className="reservation-card-actions">
                  <button
                    type="button"
                    className="btn btn-secondary reservation-btn-change"
                    onClick={() => handleChangeClick(r)}
                    disabled={!!cancellingId}
                  >
                    変更する
                  </button>
                  <button
                    type="button"
                    className="btn reservation-btn-cancel"
                    onClick={() => handleCancelClick(r)}
                    disabled={cancellingId === r.id}
                    aria-label="この予約をキャンセルする"
                  >
                    {cancellingId === r.id ? 'キャンセル中…' : 'キャンセル'}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </>
      )}

      {!loading && !error && upcoming.length === 0 && (
        <div className="reservations-empty">
          <p className="reservations-empty-text">今後の予約はありません</p>
          <button type="button" className="btn btn-primary btn-nav" onClick={() => navigate('/reserve/form')}>
            予約する
          </button>
        </div>
      )}

      {/* 予約履歴（過去） */}
      {!loading && !error && past.length > 0 && (
        <>
          <button
            type="button"
            className="btn btn-text reservations-history-toggle"
            onClick={() => setShowHistory((v) => !v)}
          >
            {showHistory ? '▲ 予約履歴を閉じる' : `▼ 予約履歴を表示（${past.length}件）`}
          </button>
          {showHistory && (
            <ul className="reservations-list reservations-list-past" aria-label="予約履歴">
              {past.map((r) => (
                <li key={r.id} className="reservation-card reservation-card-past">
                  <p className="reservation-card-datetime">
                    <span className="reservation-card-datetime-icon" aria-hidden>📅</span>
                    {formatDateTime(r.date, r.time)}
                  </p>
                  <p className="reservation-card-meta">
                    {[r.department || '—', r.purpose || '—'].filter(Boolean).join(' / ')}
                  </p>
                  <p className="reservation-card-past-label">受診済み</p>
                </li>
              ))}
            </ul>
          )}
        </>
      )}

      <div className="reservations-footer">
        <button type="button" className="btn btn-text" onClick={handleLogout}>
          ログアウト
        </button>
      </div>

      {/* キャンセル確認モーダル（Escape / バックドロップクリック対応） */}
      {cancelTarget && (
        <CancelDialog
          cancelTarget={cancelTarget}
          cancellingId={cancellingId}
          onConfirm={handleCancelConfirm}
          onClose={() => setCancelTarget(null)}
        />
      )}
    </div>
  );
}

export default MyReservationsPage;
