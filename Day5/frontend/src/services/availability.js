/**
 * 予約可否判定ロジック（診療科 × 日付 × 時間）
 * ①担当医指定時: その担当医がその日時に予約を持っていなければ可
 * ②担当医未指定時: 各診療科ごとに判定する。
 *    選択した診療科に所属する担当医の予約状況のみを確認し、
 *    いずれか1人でも空いていれば可（自動割当: ID昇順で先に空いている担当医を割り当て）
 */
import { getTimeSlots } from '../constants/masterData';
import { getWeekdayKey } from '../utils/schedule';
import { getDoctorsByDepartment } from './doctors';
import { getReservedTimesForDoctorAndDate } from './reservation';

function isValidYmd(dateStr) {
  return typeof dateStr === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(dateStr);
}

function assertQueryReady({ departmentLabel, date, time }) {
  const dept = String(departmentLabel ?? '').trim();
  if (!dept) throw new Error('[availability] departmentLabel is required');
  if (!isValidYmd(date)) throw new Error(`[availability] invalid date format (expected YYYY-MM-DD): ${String(date)}`);
  if (time != null) {
    const t = String(time).trim();
    if (!/^\d{2}:\d{2}$/.test(t)) throw new Error(`[availability] invalid time format (expected HH:mm): ${String(time)}`);
  }
  return { dept };
}

function logAvailability(event, payload) {
  if (import.meta.env.DEV) {
    // eslint-disable-next-line no-console
    console.log(`[availability] ${event}`, payload ?? '');
  }
}

/**
 * 指定した診療科・日付・時間について、空いている担当医を1名返す（ID昇順で先着）。
 * 担当医未指定で予約する際の自動割当に使用。他診療科の医師は参照しない（各診療科ごと）。
 * @param {string} departmentLabel - 診療科の表示名（その診療科のみ対象）
 * @param {string} date - YYYY-MM-DD
 * @param {string} time - 例 "09:00"
 * @returns {Promise<{ id: string, name: string } | null>} 割り当て可能な担当医、満枠なら null
 */
export async function getAvailableDoctorForSlot(departmentLabel, date, time) {
  // 「条件未確定で取得処理を走らせない」
  if (!departmentLabel || !date || !time) return null;
  assertQueryReady({ departmentLabel, date, time });

  const startedAt = Date.now();
  logAvailability('start:getAvailableDoctorForSlot', { departmentLabel, date, time });
  try {
    const doctors = await getDoctorsByDepartment(departmentLabel); // その診療科の医師のみ
    if (!Array.isArray(doctors)) throw new Error('[availability] doctors result is not an array');
    if (doctors.length === 0) {
      logAvailability('result:getAvailableDoctorForSlot', { ok: true, assigned: null, reason: 'no_doctors', ms: Date.now() - startedAt });
      return null;
    }
    const weekday = getWeekdayKey(date);
    const sorted = [...doctors].sort((a, b) => (a.id || '').localeCompare(b.id || ''));

    for (const d of sorted) {
      const scheduleSlots = d.schedules?.[weekday];
      if (!Array.isArray(scheduleSlots) || !scheduleSlots.includes(time)) continue;
      const reserved = await getReservedTimesForDoctorAndDate(departmentLabel, d.name || '', date);
      if (!(reserved instanceof Set)) throw new Error('[availability] reservedTimes result is not a Set');
      if (!reserved.has(time)) {
        const assigned = { id: d.id, name: d.name || '' };
        logAvailability('result:getAvailableDoctorForSlot', { ok: true, assigned, ms: Date.now() - startedAt });
        return assigned;
      }
    }
    logAvailability('result:getAvailableDoctorForSlot', { ok: true, assigned: null, reason: 'fully_booked_or_off_schedule', ms: Date.now() - startedAt });
    return null;
  } catch (err) {
    logAvailability('error:getAvailableDoctorForSlot', { message: err?.message, departmentLabel, date, time });
    throw err;
  }
}

/** 空き状況の短期キャッシュ（同一診療科・日付の再取得を省略） */
const availabilityCache = new Map();
const CACHE_TTL_MS = 60 * 1000; // 1分

function getCacheKey(departmentLabel, date) {
  return `${String(departmentLabel).trim()}\n${String(date).trim()}`;
}

function getCached(key) {
  const entry = availabilityCache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.at > CACHE_TTL_MS) {
    availabilityCache.delete(key);
    return null;
  }
  return entry.data;
}

function setCached(key, data) {
  availabilityCache.set(key, { at: Date.now(), data });
}

/** 予約確定・キャンセル後にキャッシュを無効化する */
export function invalidateAvailabilityCache() {
  availabilityCache.clear();
}

/**
 * 指定した診療科・日付について、全時間枠の空き状況を返す。祝日・理由はバックエンドのレスポンスのみ使用（フロントで判定しない）。
 * @param {string} departmentLabel - 診療科の表示名（その診療科のみ対象）
 * @param {string} date - YYYY-MM-DD
 * @returns {Promise<{ timeSlots: string[], availableDoctorByTime: Record<string, { id: string, name: string } | null>, isHoliday?: boolean, reason?: string | null }>}
 */
export async function getDepartmentAvailabilityForDate(departmentLabel, date, idToken) {
  const timeSlots = getTimeSlots();
  const availableDoctorByTime = {};
  if (!departmentLabel || !date) {
    timeSlots.forEach((t) => { availableDoctorByTime[t] = null; });
    return { timeSlots, availableDoctorByTime, isHoliday: false, reason: null };
  }
  assertQueryReady({ departmentLabel, date });

  const cacheKey = getCacheKey(departmentLabel, date);
  const cached = getCached(cacheKey);
  if (cached) {
    logAvailability('result:getDepartmentAvailabilityForDate', { ok: true, fromCache: true });
    return cached;
  }

  const startedAt = Date.now();
  logAvailability('start:getDepartmentAvailabilityForDate', { departmentLabel, date });
  try {
    const { getSlots } = await import('./backend');
    const result = await getSlots(departmentLabel, date, idToken);
    const slots = result.slots ?? [];
    timeSlots.forEach((t) => {
      const slot = slots.find((s) => s.time === t);
      availableDoctorByTime[t] = slot?.reservable === true ? { id: 'auto', name: '（自動割当）' } : null;
    });
    const out = {
      timeSlots,
      availableDoctorByTime,
      isHoliday: Boolean(result.isHoliday),
      reason: result.reason ?? null,
    };
    setCached(cacheKey, out);
    logAvailability('result:getDepartmentAvailabilityForDate', {
      ok: true,
      timeSlots: timeSlots.length,
      ms: Date.now() - startedAt,
      isDemoFallback: result.isDemoFallback ?? false,
      isHoliday: result.isHoliday,
    });
    return out;
  } catch (err) {
    logAvailability('error:getDepartmentAvailabilityForDate', { message: err?.message, departmentLabel, date });
    throw err;
  }
}

/**
 * 担当医×時間枠の空き状況を ○/× 用に返す（UI用）。
 * ○: 予約可能（その時間にその担当医が勤務していて、未予約）
 * ×: 予約不可（勤務外または満枠）
 * ユーザーは担当医を選ばず、空いている時間を選ぶと自動割当。UIでは「どの医師がどの時間に空いているか」を明示。
 * @param {string} departmentLabel - 診療科の表示名
 * @param {string} date - YYYY-MM-DD
 * @returns {Promise<{ doctors: Array<{id:string,name:string}>, timeSlots: string[], grid: Record<string,Record<string,'available'|'unavailable'>>, assignableByTime: Record<string,{id:string,name:string}|null> }>}
 */
export async function getDoctorTimeGrid(departmentLabel, date) {
  const timeSlots = getTimeSlots();
  // 条件未確定で取得を走らせない（UI側で idle 扱いにする）
  if (!departmentLabel || !date) {
    return { doctors: [], timeSlots: [...timeSlots], grid: {}, assignableByTime: {} };
  }
  assertQueryReady({ departmentLabel, date });

  const startedAt = Date.now();
  logAvailability('start:getDoctorTimeGrid', { departmentLabel, date });
  try {
    const doctors = await getDoctorsByDepartment(departmentLabel);
    if (!Array.isArray(doctors)) throw new Error('[availability] doctors result is not an array');
    const result = {
      doctors: doctors.map((d) => ({ id: d.id, name: d.name || '' })),
      timeSlots: [...timeSlots],
      grid: {},
      assignableByTime: {},
    };
    if (doctors.length === 0) {
      // 成功だが「医師が存在しない」
      timeSlots.forEach((t) => { result.assignableByTime[t] = null; });
      logAvailability('result:getDoctorTimeGrid', { ok: true, doctors: 0, timeSlots: timeSlots.length, ms: Date.now() - startedAt, reason: 'no_doctors' });
      return result;
    }

    const weekday = getWeekdayKey(date);
    const sorted = [...doctors].sort((a, b) => (a.id || '').localeCompare(b.id || ''));
    const reservedByDoctor = {};
    for (const d of sorted) {
      const reserved = await getReservedTimesForDoctorAndDate(departmentLabel, d.name || '', date);
      if (!(reserved instanceof Set)) throw new Error('[availability] reservedTimes result is not a Set');
      reservedByDoctor[d.id] = reserved;
    }

    for (const d of sorted) {
      result.grid[d.id] = {};
      const scheduleSlots = d.schedules?.[weekday];
      const reserved = reservedByDoctor[d.id] || new Set();
      for (const time of timeSlots) {
        const onSchedule = Array.isArray(scheduleSlots) && scheduleSlots.includes(time);
        const notReserved = !reserved.has(time);
        if (!onSchedule || !notReserved) {
          result.grid[d.id][time] = 'unavailable';
          continue;
        }
        result.grid[d.id][time] = 'available';
      }
    }

    for (const time of timeSlots) {
      let assigned = null;
      for (const d of sorted) {
        if (result.grid[d.id][time] === 'available') {
          assigned = { id: d.id, name: d.name || '' };
          break;
        }
      }
      result.assignableByTime[time] = assigned;
    }

    logAvailability('result:getDoctorTimeGrid', { ok: true, doctors: doctors.length, timeSlots: timeSlots.length, ms: Date.now() - startedAt });
    return result;
  } catch (err) {
    logAvailability('error:getDoctorTimeGrid', { message: err?.message, departmentLabel, date });
    throw err;
  }
}
