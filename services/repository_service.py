from config.s3 import get_client
from botocore.exceptions import NoCredentialsError
import hashlib
import os

S3_BUCKET = os.getenv("AWS_S3_BUCKET")
CACHE_DIR = "/app/cache_s3"  # 📌 Directory locale per la cache

class RepositoryService:
    def __init__(self):
        self.client = get_client()
        os.makedirs(CACHE_DIR, exist_ok=True)  # 📌 Assicuriamoci che la cartella di cache esista

    def get_cache_path(self, s3_key):
        """Restituisce il percorso del file cache basato sulla chiave S3."""
        # Usa la chiave S3 come parte del percorso di cache
        video_hash = hashlib.sha256(s3_key.encode()).hexdigest()  # Calcola un hash univoco per la chiave S3
        return os.path.join(CACHE_DIR, video_hash)

    def download_or_cache_video(self, s3_key, local_video_path):
        """Verifica se il video è già in cache, se no lo scarica da S3 e lo copia nella cache."""
        
        # Calcola la cache path basata sulla chiave S3
        cache_path = self.get_cache_path(s3_key)  # Cache path usando la chiave S3

        # Se il video è già presente nella cache, copialo nel percorso di elaborazione
        if os.path.exists(cache_path):
            print(f"✅ Video trovato nella cache, copiando {cache_path} a {local_video_path}")
            os.system(f"cp {cache_path} {local_video_path}")
        else:
            print(f"⬇️ Scaricamento video da S3: {s3_key}")
            # Scarica il video da S3
            self.client.download_file(S3_BUCKET, s3_key, local_video_path)
            # Copia il video scaricato nella cache per il futuro utilizzo
            os.system(f"cp {local_video_path} {cache_path}")
            
        return local_video_path



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