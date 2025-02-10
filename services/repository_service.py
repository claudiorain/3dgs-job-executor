from config.s3 import get_client
import os

s3_bucket = os.getenv("AWS_S3_BUCKET")

class RepositoryService:
    def __init__(self):
        self.client = get_client()

    def download(self, url, local_path):
        print(f"Scaricamento del file {url} da S3...")

        # 🔹 Crea la cartella se non esiste
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        self.client.download_file(s3_bucket, url, local_path)
        print(f"Download completato: {local_path}")

    def upload(self, local_path, s3_key):
        """Carica un file su S3"""
        try:
            print(f"Caricamento del file {local_path} su S3 come {s3_key}...")
            self.client.upload_file(local_path, s3_bucket, s3_key)
            print(f"Upload completato: {s3_key}")
        except Exception as e:
            print(f"Errore durante l'upload: {e}")
            return None
