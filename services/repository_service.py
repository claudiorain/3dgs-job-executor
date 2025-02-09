from config.s3 import get_client
import os

s3_bucket = os.getenv("AWS_S3_BUCKET")


class RepositoryService:
    def __init__(self):
        self.client = get_client()

    def download(self,url,local_path):
        print(f"Scaricamento del file {url} da S3...")
        self.client.download_file(s3_bucket,url, local_path)
        print(f"Download completato: {local_path}")
