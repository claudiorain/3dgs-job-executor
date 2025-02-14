from config.s3 import get_client
import os
from botocore.exceptions import NoCredentialsError

S3_BUCKET = os.getenv("AWS_S3_BUCKET")
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
            self.client.download_file(S3_BUCKET, s3_key, local_path)
            os.system(f"cp {local_path} {cache_path}")  # Salva 

    def upload(self, file_path, s3_key):
        """Carica un file su S3."""
        try:
            # Carica il file su S3
            self.client.upload_file(file_path, S3_BUCKET, s3_key)
            print(f"File caricato con successo su S3: {s3_key}")
        except FileNotFoundError:
            print(f"Errore: il file {file_path} non è stato trovato.")
        except NoCredentialsError:
            print("Errore: Credenziali AWS mancanti.")
        except Exception as e:
            print(f"Errore durante l'upload su S3: {e}")
            raise e  # Rilancia l'eccezione per essere gestita a livello superiore