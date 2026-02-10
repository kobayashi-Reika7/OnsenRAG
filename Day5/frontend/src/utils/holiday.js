/**
 * 日本の祝日判定（簡易：固定祝日＋ハッピーマンデー＋春分・秋分）
 * カレンダー・予約フォームで共通利用。
 * ※ バックエンド reservation_service._is_japanese_holiday と同一条件にすること。
 */

/** 第n月曜の日を返す (month は 0-indexed) */
function nthMonday(year, month, n) {
  const first = new Date(year, month, 1);
  // first.getDay(): 0=Sun, 1=Mon, ..., 6=Sat
  const dayOfWeek = first.getDay();
  const firstMonday = dayOfWeek <= 1 ? 1 + (1 - dayOfWeek) : 1 + (8 - dayOfWeek);
  return firstMonday + (n - 1) * 7;
}

/** 春分の日（簡易天文計算: 2000〜2099年用） */
function vernalEquinoxDay(year) {
  return Math.floor(20.8431 + 0.242194 * (year - 1980) - Math.floor((year - 1980) / 4));
}

/** 秋分の日（簡易天文計算: 2000〜2099年用） */
function autumnalEquinoxDay(year) {
  return Math.floor(23.2488 + 0.242194 * (year - 1980) - Math.floor((year - 1980) / 4));
}

/**
 * @param {Date} date
 * @returns {boolean}
 */
export function isJapaneseHoliday(date) {
  const y = date.getFullYear();
  const m = date.getMonth() + 1;
  const d = date.getDate();
  // 固定祝日
  const fixed = [
    [1, 1],    // 元日
    [2, 11],   // 建国記念の日
    [2, 23],   // 天皇誕生日（令和）
    [4, 29],   // 昭和の日
    [5, 3],    // 憲法記念日
    [5, 4],    // みどりの日
    [5, 5],    // こどもの日
    [8, 11],   // 山の日
    [11, 3],   // 文化の日
    [11, 23],  // 勤労感謝の日
  ];
  if (fixed.some(([mm, dd]) => mm === m && dd === d)) return true;
  // 成人の日（1月第2月曜）
  if (m === 1 && d === nthMonday(y, 0, 2)) return true;
  // 春分の日
  if (m === 3 && d === vernalEquinoxDay(y)) return true;
  // 海の日（7月第3月曜）
  if (m === 7 && d === nthMonday(y, 6, 3)) return true;
  // 敬老の日（9月第3月曜）
  if (m === 9 && d === nthMonday(y, 8, 3)) return true;
  // 秋分の日
  if (m === 9 && d === autumnalEquinoxDay(y)) return true;
  // スポーツの日（10月第2月曜）
  if (m === 10 && d === nthMonday(y, 9, 2)) return true;
  return false;
}
