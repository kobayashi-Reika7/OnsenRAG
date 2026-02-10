"""
予約・空き枠の業務ロジック（バックエンド専用）
- 勤務判定・空き判定・医師割当はすべてここで行う
- フロントは API の { time, reservable } のみ表示する
- 予約の正規情報は Firestore（doctorId 付き）で、キャンセル時もスロットが正しく解放される
- 環境変数 USE_DEMO_SLOTS=1 のとき、医師データが無い場合にデモ用の○を返す（シード未実行時用）
"""
from __future__ import annotations

import logging
import math
import os
import threading
from datetime import datetime
from typing import Any

from firebase_admin import firestore

logger = logging.getLogger(__name__)

from firebase_admin_client import init_firebase_admin

# 15分刻み 09:00〜16:45（フロント getTimeSlots と一致）
TIME_SLOTS = [
    f"{h:02d}:{m:02d}"
    for h in range(9, 17)
    for m in (0, 15, 30, 45)
]

WEEKDAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _get_firestore():
    init_firebase_admin()
    return firestore.client()


def _weekday_key(date_str: str) -> str:
    """YYYY-MM-DD → mon, tue, ... (0=Mon)"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return WEEKDAY_KEYS[dt.weekday()]
    except (ValueError, IndexError):
        return "sun"


def _normalize_schedules(schedules: dict[str, Any] | None) -> dict[str, list[str]]:
    """schedules を WEEKDAY_KEYS ごとの list に正規化（キー欠損・非list を [] に）"""
    schedules = schedules or {}
    out: dict[str, list[str]] = {}
    for k in WEEKDAY_KEYS:
        val = schedules.get(k)
        out[k] = list(val) if isinstance(val, list) else []
    return out


def _get_doctors_by_department(department_label: str) -> list[dict[str, Any]]:
    """Firestore doctors から診療科（表示名）で医師一覧を取得"""
    db = _get_firestore()
    coll = db.collection("doctors")
    q = coll.where("department", "==", department_label.strip())
    snap = q.stream()
    out = []
    for doc in snap:
        d = doc.to_dict()
        schedules = _normalize_schedules(d.get("schedules"))
        out.append({
            "id": doc.id,
            "name": d.get("name") or "",
            "department": d.get("department") or "",
            "schedules": schedules,
        })
    return out


def _slot_doc_id(doctor_id: str, date: str, time: str) -> str:
    """booked_slots のドキュメントID（医師+日+時間で一意）"""
    return f"{doctor_id}_{date}_{time}"


def _has_reservation(doctor_id: str, date: str, time: str) -> bool:
    """
    該当医師・日・時間のスロットが予約済みか。
    booked_slots（高速）を先にチェックし、なければ reservations（正規データ）も確認する。
    これにより booked_slots マイグレーション未実施でも正しく判定できる。
    """
    db = _get_firestore()
    # 1. booked_slots チェック（O(1) ドキュメント読み取り）
    slot_doc = db.collection("booked_slots").document(_slot_doc_id(doctor_id, date, time)).get()
    if slot_doc.exists:
        return True
    # 2. フォールバック: reservations collectionGroup チェック
    q = (db.collection_group("reservations")
         .where("doctorId", "==", doctor_id)
         .where("date", "==", date)
         .where("time", "==", time)
         .limit(1))
    return len(list(q.stream())) > 0


def _is_working(doctor: dict[str, Any], date: str, time: str) -> bool:
    """その日・その時間に勤務しているか（schedules の配列に time が含まれるか）"""
    key = _weekday_key(date)
    slots = doctor.get("schedules") or {}
    arr = slots.get(key) or []
    return time in arr


def _get_available_doctors(department_label: str, date: str, time: str) -> list[dict[str, Any]]:
    """その診療科・日・時間で空いている医師一覧（勤務かつ未予約）"""
    doctors = _get_doctors_by_department(department_label)
    return [
        d for d in doctors
        if _is_working(d, date, time) and not _has_reservation(d["id"], date, time)
    ]


def is_reservable(department_label: str, date: str, time: str) -> bool:
    """
    診療科・日・時間で予約可能か（1人でも空いていれば True）。
    get_slots（availability API）と create_reservation の両方で使用する共通判定。
    """
    return len(_get_available_doctors(department_label, date, time)) > 0


def assign_doctor(available_doctors: list[dict[str, Any]]) -> dict[str, Any] | None:
    """空いている医師から1人を選択（先頭を返す）"""
    if not available_doctors:
        return None
    return available_doctors[0]


def _nth_monday(year: int, month: int, n: int) -> int:
    """year/month の第n月曜の日を返す（weekday(): 0=Mon）"""
    first = datetime(year, month, 1)
    return (n - 1) * 7 + 1 + (7 - first.weekday()) % 7


def _vernal_equinox_day(year: int) -> int:
    """春分の日（簡易天文計算: 2000〜2099年用）"""
    return math.floor(20.8431 + 0.242194 * (year - 1980) - math.floor((year - 1980) / 4))


def _autumnal_equinox_day(year: int) -> int:
    """秋分の日（簡易天文計算: 2000〜2099年用）"""
    return math.floor(23.2488 + 0.242194 * (year - 1980) - math.floor((year - 1980) / 4))


def _is_japanese_holiday(date_str: str) -> bool:
    """日本の祝日かどうか。フロント utils/holiday.js isJapaneseHoliday と同一条件にすること。"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return False
    y, m, d = dt.year, dt.month, dt.day
    # 固定祝日
    fixed = [
        (1, 1),    # 元日
        (2, 11),   # 建国記念の日
        (2, 23),   # 天皇誕生日（令和）
        (4, 29),   # 昭和の日
        (5, 3),    # 憲法記念日
        (5, 4),    # みどりの日
        (5, 5),    # こどもの日
        (8, 11),   # 山の日
        (11, 3),   # 文化の日
        (11, 23),  # 勤労感謝の日
    ]
    if (m, d) in fixed:
        return True
    # 成人の日（1月第2月曜）
    if m == 1 and d == _nth_monday(y, 1, 2):
        return True
    # 春分の日
    if m == 3 and d == _vernal_equinox_day(y):
        return True
    # 海の日（7月第3月曜）
    if m == 7 and d == _nth_monday(y, 7, 3):
        return True
    # 敬老の日（9月第3月曜）
    if m == 9 and d == _nth_monday(y, 9, 3):
        return True
    # 秋分の日
    if m == 9 and d == _autumnal_equinox_day(y):
        return True
    # スポーツの日（10月第2月曜）
    if m == 10 and d == _nth_monday(y, 10, 2):
        return True
    return False


def _demo_reservable(date_str: str, time_str: str) -> bool:
    """デモ用: 平日の 09:00〜11:45 を予約可とする（シード未実行時用）"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if dt.weekday() >= 5:  # 土日は×
            return False
    except (ValueError, TypeError):
        return False
    demo_times = [t for t in TIME_SLOTS if t < "12:00"]  # 09:00〜11:45
    return time_str in demo_times


def get_slots(department_label: str, date: str) -> list[dict[str, str | bool]]:
    """
    診療科・日付に対する全時間枠の予約可否を返す。
    内部で get_availability_for_dates を使い、Firestore クエリ2回で完結（N+1 解消）。
    """
    if not department_label or not date:
        return [{"time": t, "reservable": False} for t in TIME_SLOTS]
    try:
        results = get_availability_for_dates(department_label, [date])
        if results and len(results) > 0:
            return results[0].get("slots", [{"time": t, "reservable": False} for t in TIME_SLOTS])
    except Exception:
        logger.exception("get_slots failed for %s / %s", department_label, date)
    return [{"time": t, "reservable": False} for t in TIME_SLOTS]


def _get_reservations_bulk(doctor_ids: list[str], dates: list[str]) -> set[tuple[str, str, str]]:
    """
    複数医師・複数日付の予約済みスロットを一括取得し、(doctorId, date, time) の set を返す。
    booked_slots と reservations の両方を確認しマージする。
    これにより booked_slots マイグレーション未実施でも正しく判定できる。
    """
    if not doctor_ids or not dates:
        return set()
    db = _get_firestore()
    reserved: set[tuple[str, str, str]] = set()
    doctor_id_set = set(doctor_ids)
    chunk_size = 30

    for i in range(0, len(dates), chunk_size):
        date_chunk = dates[i:i + chunk_size]

        # 1. booked_slots から取得
        try:
            q = db.collection("booked_slots").where("date", "in", date_chunk)
            for doc in q.stream():
                d = doc.to_dict()
                did = d.get("doctorId", "")
                if did in doctor_id_set:
                    reserved.add((did, d.get("date", ""), d.get("time", "")))
        except Exception as e:
            logger.warning("_get_reservations_bulk booked_slots failed: %s", e)

        # 2. reservations collectionGroup からも取得（フォールバック）
        try:
            q = db.collection_group("reservations").where("date", "in", date_chunk)
            for doc in q.stream():
                d = doc.to_dict()
                did = d.get("doctorId", "")
                if did in doctor_id_set:
                    reserved.add((did, d.get("date", ""), d.get("time", "")))
        except Exception as e:
            logger.warning("_get_reservations_bulk reservations failed: %s", e)

    return reserved


def _get_user_reservations_for_dates(user_id: str, department_label: str, dates: list[str]) -> set[tuple[str, str]]:
    """
    指定ユーザーが指定診療科・指定日付で既に予約している (date, time) のセットを返す。
    同一ユーザーの重複予約を防ぐために使用。
    """
    if not user_id or not department_label or not dates:
        return set()
    db = _get_firestore()
    booked: set[tuple[str, str]] = set()
    chunk_size = 30
    for i in range(0, len(dates), chunk_size):
        date_chunk = dates[i:i + chunk_size]
        try:
            ref = db.collection("users").document(user_id).collection("reservations")
            q = ref.where("department", "==", department_label).where("date", "in", date_chunk)
            for doc in q.stream():
                d = doc.to_dict()
                dt = d.get("date", "")
                tm = d.get("time", "")
                if dt and tm:
                    booked.add((dt, tm))
        except Exception as e:
            logger.warning("_get_user_reservations_for_dates failed: %s", e)
    return booked


def get_availability_for_dates(department_label: str, dates: list[str], *, user_id: str = "") -> list[dict[str, Any]]:
    """
    複数日分の空き状況を一括で返す（高速版）。
    医師取得1回 + 予約取得1回 = Firestore 2クエリで全日分を計算。
    user_id が指定された場合、そのユーザーが既に予約済みのスロットも reservable=False にする。
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    use_demo = os.environ.get("USE_DEMO_SLOTS", "1").strip() != "0"
    all_false = [{"time": t, "reservable": False} for t in TIME_SLOTS]

    # 過去日・祝日は即決定
    results: dict[str, dict[str, Any]] = {}
    dates_to_compute: list[str] = []
    for date in dates:
        if not date or not department_label:
            results[date] = {"date": date or "", "is_holiday": False, "reservable": False, "reason": "closed", "slots": all_false}
            continue
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            if dt.date() < today.date():
                results[date] = {"date": date, "is_holiday": False, "reservable": False, "reason": "past", "slots": all_false}
                continue
        except (ValueError, TypeError):
            results[date] = {"date": date, "is_holiday": False, "reservable": False, "reason": "closed", "slots": all_false}
            continue
        if _is_japanese_holiday(date):
            results[date] = {"date": date, "is_holiday": True, "reservable": False, "reason": "holiday", "slots": all_false}
            continue
        dates_to_compute.append(date)

    # 計算対象の日付がなければ即返却
    if not dates_to_compute:
        return [results[d] for d in dates]

    # 医師取得（1回のみ）
    try:
        doctors = _get_doctors_by_department(department_label)
    except Exception:
        doctors = []

    if not doctors and use_demo:
        for date in dates_to_compute:
            slot_list = [{"time": t, "reservable": _demo_reservable(date, t)} for t in TIME_SLOTS]
            any_ok = any(s["reservable"] for s in slot_list)
            results[date] = {"date": date, "is_holiday": False, "reservable": any_ok, "reason": None, "slots": slot_list}
        return [results[d] for d in dates]

    if not doctors:
        for date in dates_to_compute:
            results[date] = {"date": date, "is_holiday": False, "reservable": False, "reason": None, "slots": all_false}
        return [results[d] for d in dates]

    # 予約一括取得（1回のみ）
    doctor_ids = [doc["id"] for doc in doctors]
    doctor_id_set = set(doctor_ids)
    try:
        reserved = _get_reservations_bulk(doctor_ids, dates_to_compute)
    except Exception as e:
        logger.warning("get_availability_for_dates: bulk reservation fetch failed: %s", e)
        reserved = set()

    # ユーザーの既存予約を取得（同一ユーザーが同じ診療科+日+時間を二重予約するのを防止）
    user_booked: set[tuple[str, str]] = set()
    if user_id:
        try:
            user_booked = _get_user_reservations_for_dates(user_id, department_label, dates_to_compute)
        except Exception as e:
            logger.warning("get_availability_for_dates: user reservation fetch failed: %s", e)

    # メモリ上で各日・各時間の空き判定
    # ルール: 勤務中かつ未予約の医師が1人でもいれば○（ただしユーザー既存予約は×）
    for date in dates_to_compute:
        slot_list = []
        for t in TIME_SLOTS:
            # このユーザーが既に同じ診療科+日+時間で予約済みなら×
            if (date, t) in user_booked:
                slot_list.append({"time": t, "reservable": False})
                continue
            available = False
            for doc in doctors:
                if _is_working(doc, date, t) and (doc["id"], date, t) not in reserved:
                    available = True
                    break
            slot_list.append({"time": t, "reservable": available})
        any_ok = any(s["reservable"] for s in slot_list)
        results[date] = {"date": date, "is_holiday": False, "reservable": any_ok, "reason": None, "slots": slot_list}

    return [results[d] for d in dates]


def get_availability_for_date(department_label: str, date: str, *, user_id: str = "") -> dict[str, Any]:
    """
    1日分の空き状況を返す。祝日・過去日はバックエンドで判定し、レスポンスに含める。
    判定優先: 過去日 → 祝日 → 休診 → 医師勤務なし → 可。
    user_id が指定された場合、そのユーザーの予約済みスロットも reservable=False にする。
    """
    slots = [{"time": t, "reservable": False} for t in TIME_SLOTS]
    if not department_label or not date:
        return {
            "date": date or "",
            "is_holiday": False,
            "reservable": False,
            "reason": "closed",
            "slots": slots,
        }
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if dt.date() < today.date():
            return {
                "date": date,
                "is_holiday": False,
                "reservable": False,
                "reason": "past",
                "slots": slots,
            }
    except (ValueError, TypeError):
        return {
            "date": date,
            "is_holiday": False,
            "reservable": False,
            "reason": "closed",
            "slots": slots,
        }
    # 祝日は必ず is_holiday: True と全枠 reservable: False を返す（フロントは isHoliday で表示制御）
    if _is_japanese_holiday(date):
        return {
            "date": date,
            "is_holiday": True,
            "reservable": False,
            "reason": "holiday",
            "slots": slots,
        }
    # user_id を伝播するため get_availability_for_dates を直接使用
    results = get_availability_for_dates(department_label, [date], user_id=user_id)
    if results:
        return results[0]
    return {
        "date": date,
        "is_holiday": False,
        "reservable": False,
        "reason": None,
        "slots": [{"time": t, "reservable": False} for t in TIME_SLOTS],
    }


# --------------- ダブルブッキング防止: スロット単位のロック ---------------
_booking_locks: dict[str, threading.Lock] = {}
_lock_manager = threading.Lock()


def _get_booking_lock(key: str) -> threading.Lock:
    """予約スロット単位のロックを取得（同一プロセス内でのダブルブッキングを防止）"""
    with _lock_manager:
        if key not in _booking_locks:
            _booking_locks[key] = threading.Lock()
        return _booking_locks[key]


def create_reservation(department_label: str, date: str, time: str, user_id: str, *, purpose: str = "") -> dict[str, Any]:
    """
    予約を確定する。担当医は自動割当。
    Firestore users/{uid}/reservations に doctorId 付きで保存。
    スロット単位のロックでダブルブッキングを防止する。
    """
    logger.info(
        "create_reservation start: department=%r date=%r time=%r user_id=%r",
        department_label, date, time, user_id,
    )
    department_label = (department_label or "").strip()
    date = (date or "").strip()
    time = (time or "").strip()
    user_id = (user_id or "").strip()
    if not user_id:
        raise ValueError("ユーザーIDが取得できません。再ログインしてください。")
    if not department_label or not date or not time:
        raise ValueError("診療科・日付・時間は必須です。")

    # 過去日チェック
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if dt.date() < today.date():
            raise ValueError("過去の日付は予約できません。別の日をお選びください。")
    except ValueError as e:
        if "過去の日付" in str(e):
            raise
    except TypeError:
        pass
    if _is_japanese_holiday(date):
        raise ValueError("祝日のため予約できません。別の日をお選びください。")

    # 今日の過去時刻チェック
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        now_str = datetime.now().strftime("%H:%M")
        if dt.date() == today.date() and time <= now_str:
            raise ValueError("この時刻はすでに過ぎています。別の時間をお選びください。")
    except ValueError as e:
        if "すでに過ぎています" in str(e):
            raise
    except TypeError:
        pass

    logger.info("create_reservation passed validation")

    # ダブルブッキング防止: プロセス内ロック + Firestore 原子的スロット確保の2段構え
    lock_key = f"{department_label}::{date}::{time}"
    slot_lock = _get_booking_lock(lock_key)
    if not slot_lock.acquire(timeout=5):
        raise ValueError("この時間は現在処理中です。しばらくしてから再度お試しください。")

    try:
        db = _get_firestore()
        use_demo = os.environ.get("USE_DEMO_SLOTS", "1").strip() != "0"

        # 同一ユーザーが同じ診療科+日+時間で既に予約しているか確認
        try:
            existing = list(
                db.collection("users").document(user_id).collection("reservations")
                .where("department", "==", department_label)
                .where("date", "==", date)
                .where("time", "==", time)
                .limit(1)
                .stream()
            )
            if existing:
                raise ValueError("この診療科・日時はすでに予約済みです。予約一覧からご確認ください。")
        except ValueError:
            raise
        except Exception as e:
            logger.warning("create_reservation user duplicate check failed: %s", e)

        try:
            available = _get_available_doctors(department_label, date, time)
        except Exception as e:
            logger.exception("create_reservation _get_available_doctors failed: %s", e)
            raise

        # --- Firestore 原子的スロット確保 ---
        # 各候補医師について booked_slots/{doctorId}_{date}_{time} を create() で確保。
        # create() はドキュメントが既に存在する場合に例外を投げるため、
        # 同時リクエストでも1つだけが成功する（Firestore レベルのアトミック保証）。
        doctor = None
        doctor_id = ""
        doctor_name = ""
        slot_doc_id = ""

        for candidate in available:
            cand_id = str(candidate.get("id") or "").strip()
            cand_name = str(candidate.get("name") or "（自動割当）").strip()
            if not cand_id:
                continue
            sid = _slot_doc_id(cand_id, date, time)
            slot_ref = db.collection("booked_slots").document(sid)
            try:
                slot_ref.create({
                    "doctorId": cand_id,
                    "date": date,
                    "time": time,
                    "department": department_label,
                    "userId": user_id,
                    "createdAt": firestore.SERVER_TIMESTAMP,
                })
                # create() が成功 = このスロットを確保できた
                doctor = candidate
                doctor_id = cand_id
                doctor_name = cand_name
                slot_doc_id = sid
                logger.info("Slot locked: %s", sid)
                break
            except Exception as e:
                # ALREADY_EXISTS = 他のリクエストが先に確保済み → 次の医師を試す
                err_str = str(e).lower()
                if "already exists" in err_str or "already_exists" in err_str or "409" in err_str:
                    logger.info("Slot %s already taken, trying next doctor", sid)
                    continue
                logger.exception("Unexpected error creating slot %s: %s", sid, e)
                raise

        # デモモードのフォールバック
        if not doctor and use_demo and _demo_reservable(date, time):
            doctor_id = "demo"
            doctor_name = "（自動割当）"
            doctor = {"id": "demo", "name": doctor_name}

        if not doctor:
            raise ValueError("この時間は現在予約できません。別の時間をお選びください。")

        # --- ユーザーの予約ドキュメントを作成 ---
        payload = {
            "date": str(date),
            "time": str(time),
            "category": "",
            "department": str(department_label),
            "purpose": str(purpose).strip(),
            "doctor": doctor_name,
            "doctorId": doctor_id,
            "createdAt": firestore.SERVER_TIMESTAMP,
        }

        try:
            ref = db.collection("users").document(user_id).collection("reservations")
            _, doc_ref = ref.add(payload)
            doc_id = doc_ref.id if doc_ref else ""
        except Exception as e:
            # 予約作成に失敗した場合、確保したスロットを解放
            if slot_doc_id:
                try:
                    db.collection("booked_slots").document(slot_doc_id).delete()
                    logger.info("Released slot %s due to reservation creation failure", slot_doc_id)
                except Exception:
                    logger.exception("Failed to release slot %s", slot_doc_id)
            logger.exception("create_reservation Firestore add failed: %s", e)
            raise

        # スロットに予約IDを記録（キャンセル時の参照用）
        if slot_doc_id:
            try:
                db.collection("booked_slots").document(slot_doc_id).update({
                    "reservationId": doc_id,
                })
            except Exception:
                logger.warning("Failed to update slot %s with reservationId", slot_doc_id)

        logger.info("create_reservation done: doc_id=%s slot=%s", doc_id, slot_doc_id)
        return {
            "id": doc_id,
            "departmentId": department_label,
            "doctorId": doctor_id,
            "date": date,
            "time": time,
            "userId": user_id,
        }
    finally:
        slot_lock.release()


def cancel_reservation(user_id: str, reservation_id: str) -> dict[str, Any]:
    """
    予約をキャンセルする。
    1. users/{uid}/reservations/{id} を読み取り、doctorId/date/time を取得
    2. booked_slots/{doctorId}_{date}_{time} を削除（スロット解放）
    3. users/{uid}/reservations/{id} を削除
    """
    if not user_id or not reservation_id:
        raise ValueError("ユーザーIDまたは予約IDが不正です。")

    db = _get_firestore()
    res_ref = db.collection("users").document(user_id).collection("reservations").document(reservation_id)
    res_doc = res_ref.get()

    if not res_doc.exists:
        raise ValueError("指定された予約が見つかりません。")

    data = res_doc.to_dict() or {}
    doctor_id = data.get("doctorId", "")
    date = data.get("date", "")
    time_val = data.get("time", "")

    # booked_slots のスロットを解放
    if doctor_id and date and time_val and doctor_id != "demo":
        sid = _slot_doc_id(doctor_id, date, time_val)
        try:
            db.collection("booked_slots").document(sid).delete()
            logger.info("cancel_reservation: released slot %s", sid)
        except Exception as e:
            logger.warning("cancel_reservation: failed to release slot %s: %s", sid, e)

    # 予約ドキュメントを削除
    res_ref.delete()
    logger.info("cancel_reservation done: user=%s reservation=%s", user_id, reservation_id)
    return {"ok": True, "id": reservation_id}
