/**
 * トップページ（ログイン前の公開ページ）
 * さくら総合病院の案内・診療科一覧（表示のみ）・診療時間・Web予約入口
 */
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../App';
import DepartmentListSelector from '../components/DepartmentListSelector';

const TOP_IMAGE = '/ヘッダー2.jpg';

const HOSPITAL_NAME = 'さくら総合病院';
const CATCH_COPY = '地域の皆様の健やかな暮らしを支えます';
const INTRO_LINES = ['内科からリハビリまで幅広い診療科を備えた総合病院です。', '安心してご来院ください。'];
const TOP_LEAD = '診療科・日時を選んで、かんたんにWeb予約ができます。';


function TopPage() {
  const navigate = useNavigate();
  const user = useAuth();

  const HOURS = [
    { label: 'Web予約枠', time: '9:00〜17:00' },
    { label: '休診日', time: '土日祝' },
    { label: '備考', time: '診療科・担当医の勤務状況により、表示される枠が異なります。' },
  ];

  const handleReserve = () => {
    if (user) {
      navigate('/menu', { replace: true });
    } else {
      navigate('/login', { replace: true });
    }
  };

  return (
    <div className="page top-page">
      <header className="top-header">
        <div className="top-hero">
          <img src={TOP_IMAGE} alt={HOSPITAL_NAME} className="top-hero-img" />
        </div>
        <h1 className="top-title">{HOSPITAL_NAME}</h1>
        <p className="top-catch">{CATCH_COPY}</p>
        <p className="top-lead">{TOP_LEAD}</p>
      </header>

      <section className="top-intro">
        <p className="top-intro-text">{INTRO_LINES[0]}<br />{INTRO_LINES[1]}</p>
      </section>

      <section className="top-section" aria-labelledby="dept-list-title">
        <h2 id="dept-list-title" className="top-section-title">
          <span className="top-section-icon" aria-hidden>📋</span>
          診療科一覧
        </h2>
        <div className="top-dept-list">
          <DepartmentListSelector />
        </div>
      </section>

      <section className="top-section">
        <h2 className="top-section-title">
          <span className="top-section-icon" aria-hidden>⏰</span>
          診療時間
        </h2>
        <dl className="top-hours">
          {HOURS.map((h) => (
            <React.Fragment key={h.label}>
              <dt className="top-hours-dt">{h.label}</dt>
              <dd className="top-hours-dd">{h.time}</dd>
            </React.Fragment>
          ))}
        </dl>
      </section>

      <footer className="top-footer">
        <button
          type="button"
          className="btn btn-primary btn-nav top-reserve-btn"
          onClick={handleReserve}
        >
          {user ? '予約メニューへ' : 'Web予約はこちら（ログイン）'}
        </button>
      </footer>
    </div>
  );
}

export default TopPage;
