"""List blobs in gs://myocr_1/output/"""
from google.cloud import storage

client = storage.Client()
bucket = client.bucket("myocr_1")

for blob in bucket.list_blobs(prefix="output/"):
    print(blob.name)
