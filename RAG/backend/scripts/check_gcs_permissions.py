"""
GCS permission check script.
Verifies service account has storage.objects.get and storage.objects.create
on bucket myocr_1.

Run:
  $env:GOOGLE_APPLICATION_CREDENTIALS="path/to/onsen.json"
  python check_gcs_permissions.py
"""
import os
from google.cloud import storage
from google.api_core import exceptions

BUCKET_NAME = "myocr_1"
TEST_OBJECT = "有馬マップ.pdf"  # 存在確認用
TEST_WRITE = "output/_permission_test.txt"  # 書き込みテスト用

def main():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print("[ERROR] GOOGLE_APPLICATION_CREDENTIALS が設定されていません")
        return

    print(f"認証: {creds_path}")
    print(f"バケット: gs://{BUCKET_NAME}")
    print("-" * 50)

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    # 1. オブジェクトの一覧（storage.objects.list）
    try:
        blobs = list(bucket.list_blobs(max_results=5))
        print(f"[OK] storage.objects.list - 一覧取得可能 ({len(blobs)}件)")
    except exceptions.Forbidden as e:
        print(f"[NG] storage.objects.list - 権限なし: {e}")
        return
    except Exception as e:
        print(f"[NG] エラー: {e}")
        return

    # 3. オブジェクトの読み取り（storage.objects.get）
    try:
        blob = bucket.blob(TEST_OBJECT)
        blob.reload()
        print(f"[OK] storage.objects.get - {TEST_OBJECT} にアクセス可能")
    except exceptions.NotFound:
        print(f"[INFO] {TEST_OBJECT} は存在しません（get権限は別オブジェクトで要確認）")
    except exceptions.Forbidden as e:
        print(f"[NG] storage.objects.get - 権限なし: {e}")

    # 4. オブジェクトの書き込み（storage.objects.create）
    try:
        test_blob = bucket.blob(TEST_WRITE)
        test_blob.upload_from_string("permission test", content_type="text/plain")
        test_blob.delete()
        print(f"[OK] storage.objects.create - output/ への書き込み可能")
    except exceptions.Forbidden as e:
        print(f"[NG] storage.objects.create - 権限なし: {e}")
    except Exception as e:
        print(f"[NG] 書き込みテスト失敗: {e}")

    print("-" * 50)
    print("チェック完了")

if __name__ == "__main__":
    main()
