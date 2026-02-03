"""
メモリ上のデータ保持（学習用・再起動でリセット）
タスクとリストを in-memory で管理する
"""
from typing import List, Optional

# リスト: id, name
lists_db: List[dict] = [
    {"id": 1, "name": "デフォルトリスト"}
]
next_list_id = 2

# タスク: id, title, list_id, is_completed, is_favorite, due_date, memo, time
tasks_db: List[dict] = []
next_task_id = 1


def get_all_tasks() -> List[dict]:
    return list(tasks_db)


def get_task_by_id(task_id: int) -> Optional[dict]:
    for t in tasks_db:
        if t["id"] == task_id:
            return t
    return None


def add_task(data: dict) -> dict:
    global next_task_id
    task = {
        "id": next_task_id,
        "title": data["title"],
        "list_id": data.get("list_id", 1),
        "is_completed": data.get("is_completed", False),
        "is_favorite": data.get("is_favorite", False),
        "due_date": data.get("due_date"),
        "memo": data.get("memo", ""),
        "time": data.get("time", 0),
    }
    tasks_db.append(task)
    next_task_id += 1
    return task


def update_task(task_id: int, data: dict) -> Optional[dict]:
    task = get_task_by_id(task_id)
    if not task:
        return None
    for key, value in data.items():
        if value is not None and key in task:
            task[key] = value
    return task


def delete_task(task_id: int) -> bool:
    global tasks_db
    for i, t in enumerate(tasks_db):
        if t["id"] == task_id:
            tasks_db.pop(i)
            return True
    return False


def get_all_lists() -> List[dict]:
    return list(lists_db)


def get_list_by_id(list_id: int) -> Optional[dict]:
    for L in lists_db:
        if L["id"] == list_id:
            return L
    return None


def add_list(name: str) -> dict:
    global next_list_id
    lst = {"id": next_list_id, "name": name}
    lists_db.append(lst)
    next_list_id += 1
    return lst


def delete_list(list_id: int) -> bool:
    global lists_db, tasks_db
    if list_id == 1:
        return False  # デフォルトリストは削除不可
    for i, L in enumerate(lists_db):
        if L["id"] == list_id:
            lists_db.pop(i)
            for t in tasks_db:
                if t["list_id"] == list_id:
                    t["list_id"] = 1
            return True
    return False
