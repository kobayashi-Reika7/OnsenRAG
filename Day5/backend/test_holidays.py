"""
祝日判定・モデルバリデーションの基本テスト
実行: cd Day5/backend && python -m pytest test_holidays.py -v
"""
import math
import pytest
from datetime import datetime


# --- reservation_service.py から祝日関連関数をコピー（firebase_admin 依存を避ける） ---
def _nth_monday(year: int, month: int, n: int) -> int:
    first = datetime(year, month, 1)
    return (n - 1) * 7 + 1 + (7 - first.weekday()) % 7

def _vernal_equinox_day(year: int) -> int:
    return math.floor(20.8431 + 0.242194 * (year - 1980) - math.floor((year - 1980) / 4))

def _autumnal_equinox_day(year: int) -> int:
    return math.floor(23.2488 + 0.242194 * (year - 1980) - math.floor((year - 1980) / 4))

def _is_japanese_holiday(date_str: str) -> bool:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return False
    y, m, d = dt.year, dt.month, dt.day
    fixed = [
        (1, 1), (2, 11), (2, 23), (4, 29), (5, 3), (5, 4), (5, 5),
        (8, 11), (11, 3), (11, 23),
    ]
    if (m, d) in fixed:
        return True
    if m == 1 and d == _nth_monday(y, 1, 2):
        return True
    if m == 3 and d == _vernal_equinox_day(y):
        return True
    if m == 7 and d == _nth_monday(y, 7, 3):
        return True
    if m == 9 and d == _nth_monday(y, 9, 3):
        return True
    if m == 9 and d == _autumnal_equinox_day(y):
        return True
    if m == 10 and d == _nth_monday(y, 10, 2):
        return True
    return False


class TestNthMonday:
    """第n月曜日の計算テスト"""

    def test_2nd_monday_jan_2026(self):
        # 2026年1月: 1日=木曜 → 第2月曜=12日
        assert _nth_monday(2026, 1, 2) == 12

    def test_3rd_monday_jul_2026(self):
        # 2026年7月: 1日=水曜 → 第3月曜=20日
        assert _nth_monday(2026, 7, 3) == 20

    def test_2nd_monday_oct_2026(self):
        # 2026年10月: 1日=木曜 → 第2月曜=12日
        assert _nth_monday(2026, 10, 2) == 12


class TestJapaneseHoliday:
    """祝日判定テスト"""

    # 固定祝日
    def test_new_years_day(self):
        assert _is_japanese_holiday("2026-01-01") is True

    def test_emperors_birthday(self):
        assert _is_japanese_holiday("2026-02-23") is True

    def test_showa_day(self):
        assert _is_japanese_holiday("2026-04-29") is True

    def test_constitution_day(self):
        assert _is_japanese_holiday("2026-05-03") is True

    def test_mountain_day(self):
        assert _is_japanese_holiday("2026-08-11") is True

    def test_culture_day(self):
        assert _is_japanese_holiday("2026-11-03") is True

    def test_labor_thanksgiving(self):
        assert _is_japanese_holiday("2026-11-23") is True

    # 令和以前の12/23は祝日ではない
    def test_dec_23_not_holiday(self):
        assert _is_japanese_holiday("2026-12-23") is False

    # ハッピーマンデー
    def test_coming_of_age_day_2026(self):
        # 2026年 成人の日 = 1月第2月曜 = 1/12
        assert _is_japanese_holiday("2026-01-12") is True

    def test_marine_day_2026(self):
        # 2026年 海の日 = 7月第3月曜 = 7/20
        assert _is_japanese_holiday("2026-07-20") is True
        # 7/18は祝日ではない（旧ロジックのバグ修正確認）
        assert _is_japanese_holiday("2026-07-18") is False

    def test_respect_for_aged_day_2026(self):
        # 2026年 敬老の日 = 9月第3月曜 = 9/21
        assert _is_japanese_holiday("2026-09-21") is True

    def test_sports_day_2026(self):
        # 2026年 スポーツの日 = 10月第2月曜 = 10/12
        assert _is_japanese_holiday("2026-10-12") is True

    # 春分・秋分
    def test_vernal_equinox_2026(self):
        assert _is_japanese_holiday("2026-03-20") is True

    def test_autumnal_equinox_2026(self):
        assert _is_japanese_holiday("2026-09-23") is True

    # 通常の平日
    def test_regular_weekday(self):
        assert _is_japanese_holiday("2026-02-10") is False

    # 無効な日付
    def test_invalid_date(self):
        assert _is_japanese_holiday("invalid") is False
        assert _is_japanese_holiday("") is False


class TestModelValidation:
    """Pydanticモデルのバリデーションテスト（pydanticがインストール済みの場合のみ実行）"""

    @pytest.fixture(autouse=True)
    def _require_pydantic(self):
        pytest.importorskip("pydantic")

    def test_valid_reservation_body(self):
        from models import CreateReservationBody
        body = CreateReservationBody(department="内科", date="2026-02-15", time="09:00")
        assert body.department == "内科"

    def test_invalid_date_format(self):
        from models import CreateReservationBody
        with pytest.raises(Exception):
            CreateReservationBody(department="内科", date="2026/02/15", time="09:00")

    def test_invalid_time_format(self):
        from models import CreateReservationBody
        with pytest.raises(Exception):
            CreateReservationBody(department="内科", date="2026-02-15", time="9:00")

    def test_empty_department(self):
        from models import CreateReservationBody
        with pytest.raises(Exception):
            CreateReservationBody(department="", date="2026-02-15", time="09:00")

    def test_department_too_long(self):
        from models import CreateReservationBody
        with pytest.raises(Exception):
            CreateReservationBody(department="a" * 101, date="2026-02-15", time="09:00")
