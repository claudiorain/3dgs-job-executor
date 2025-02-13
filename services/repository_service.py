from config.s3 import get_client
import os

s3_bucket = os.getenv("AWS_S3_BUCKET")
CACHE_DIR = "/app/cache_s3"  # 📌 Directory locale per la cache

class RepositoryService:
    def __init__(self):
        self.client = get_client()
        os.makedirs(CACHE_DIR, exist_ok=True)  # 📌 Assicuriamoci che la cartella di cache esista

    def get_cache_path(self, s3_key):
        """ Restituisce il percorso locale nella cache per un dato file """
        return os.path.join(CACHE_DIR, s3_key.replace("/", "_"))

    def download(self, s3_key, local_path):
        """ Scarica un file da S3 solo se non è già presente in cache """
        cache_path = self.get_cache_path(s3_key)

        if os.path.exists(cache_path):
            print(f"✅ Usando il file in cache: {cache_path}")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            os.system(f"cp {cache_path} {local_path}")  # Copia il file dalla cache
        else:
            print(f"⬇️ Scaricamento del file {s3_key} da S3...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.client.download_file(s3_bucket, s3_key, local_path)
            os.system(f"cp {local_path} {cache_path}")  # Salva 
