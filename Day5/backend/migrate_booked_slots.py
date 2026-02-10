"""
既存の予約データから booked_slots コレクションを構築するマイグレーションスクリプト。
初回のみ実行してください。2回目以降は冪等（既存ドキュメントはスキップ）です。

使い方:
  cd Day5/backend
  python migrate_booked_slots.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from firebase_admin import firestore as fs
from firebase_admin_client import init_firebase_admin


def migrate():
    init_firebase_admin()
    db = fs.client()

    # collectionGroup("reservations") で全予約を取得
    logger.info("Fetching all reservations via collectionGroup...")
    reservations = list(db.collection_group("reservations").stream())
    logger.info("Found %d reservations", len(reservations))

    created = 0
    skipped = 0
    errors = 0

    for doc in reservations:
        data = doc.to_dict()
        doctor_id = data.get("doctorId", "")
        date = data.get("date", "")
        time_val = data.get("time", "")
        department = data.get("department", "")

        if not doctor_id or not date or not time_val:
            logger.warning("Skipping reservation %s: missing doctorId/date/time", doc.id)
            skipped += 1
            continue

        if doctor_id == "demo":
            skipped += 1
            continue

        # ユーザーIDをパスから取得: users/{uid}/reservations/{id}
        path_parts = doc.reference.path.split("/")
        user_id = path_parts[1] if len(path_parts) >= 2 else ""

        slot_id = f"{doctor_id}_{date}_{time_val}"
        slot_ref = db.collection("booked_slots").document(slot_id)

        try:
            slot_ref.create({
                "doctorId": doctor_id,
                "date": date,
                "time": time_val,
                "department": department,
                "userId": user_id,
                "reservationId": doc.id,
                "createdAt": fs.SERVER_TIMESTAMP,
            })
            created += 1
            logger.info("Created booked_slot: %s", slot_id)
        except Exception as e:
            err_str = str(e).lower()
            if "already exists" in err_str or "already_exists" in err_str:
                skipped += 1
                logger.info("Slot already exists: %s", slot_id)
            else:
                errors += 1
                logger.error("Failed to create slot %s: %s", slot_id, e)

    logger.info("Migration complete: created=%d, skipped=%d, errors=%d", created, skipped, errors)


if __name__ == "__main__":
    migrate()
