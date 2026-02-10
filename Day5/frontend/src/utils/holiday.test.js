/**
 * 祝日判定テスト
 * 実行: cd Day5/frontend && npm test
 */
import { describe, it, expect } from 'vitest';
import { isJapaneseHoliday } from './holiday';

describe('isJapaneseHoliday', () => {
  // 固定祝日
  it('元日', () => expect(isJapaneseHoliday(new Date(2026, 0, 1))).toBe(true));
  it('建国記念の日', () => expect(isJapaneseHoliday(new Date(2026, 1, 11))).toBe(true));
  it('天皇誕生日（令和）', () => expect(isJapaneseHoliday(new Date(2026, 1, 23))).toBe(true));
  it('昭和の日', () => expect(isJapaneseHoliday(new Date(2026, 3, 29))).toBe(true));
  it('憲法記念日', () => expect(isJapaneseHoliday(new Date(2026, 4, 3))).toBe(true));
  it('みどりの日', () => expect(isJapaneseHoliday(new Date(2026, 4, 4))).toBe(true));
  it('こどもの日', () => expect(isJapaneseHoliday(new Date(2026, 4, 5))).toBe(true));
  it('山の日', () => expect(isJapaneseHoliday(new Date(2026, 7, 11))).toBe(true));
  it('文化の日', () => expect(isJapaneseHoliday(new Date(2026, 10, 3))).toBe(true));
  it('勤労感謝の日', () => expect(isJapaneseHoliday(new Date(2026, 10, 23))).toBe(true));

  // 令和以前の12/23は祝日ではない
  it('12/23は令和では祝日でない', () => expect(isJapaneseHoliday(new Date(2026, 11, 23))).toBe(false));

  // ハッピーマンデー
  it('成人の日 2026 = 1/12', () => expect(isJapaneseHoliday(new Date(2026, 0, 12))).toBe(true));
  it('海の日 2026 = 7/20', () => expect(isJapaneseHoliday(new Date(2026, 6, 20))).toBe(true));
  it('7/18は海の日ではない', () => expect(isJapaneseHoliday(new Date(2026, 6, 18))).toBe(false));
  it('敬老の日 2026 = 9/21', () => expect(isJapaneseHoliday(new Date(2026, 8, 21))).toBe(true));
  it('スポーツの日 2026 = 10/12', () => expect(isJapaneseHoliday(new Date(2026, 9, 12))).toBe(true));

  // 春分・秋分
  it('春分の日 2026 = 3/20', () => expect(isJapaneseHoliday(new Date(2026, 2, 20))).toBe(true));
  it('秋分の日 2026 = 9/23', () => expect(isJapaneseHoliday(new Date(2026, 8, 23))).toBe(true));

  // 通常の平日
  it('平日は祝日でない', () => expect(isJapaneseHoliday(new Date(2026, 1, 10))).toBe(false));
});
