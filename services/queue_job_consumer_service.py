from config.message_queue import get_connection  # Assicurati che questa funzione restituisca il client del database
from config.message_queue import get_channel  # Assicurati che questa funzione restituisca il client del database
from config.message_queue import close_connection  # Assicurati che questa funzione restituisca il client del database
from services.model_service import ModelService
from services.repository_service import RepositoryService
import os
import sys
import json
import asyncio
import requests
import cv2
import shutil
import numpy as np

model_service = ModelService()
repository_service = RepositoryService()

WORKING_DIR = os.getenv("MODEL_WORKING_DIR") 
GAUSSIAN_SPLATTING_API_URL = "http://gaussian-splatting-api:8050"

# Assicurati che la cartella esista
os.makedirs(WORKING_DIR, exist_ok=True)

class QueueJobService:

    def __init__(self):
        """Inizializza la connessione a RabbitMQ (può rimanere None finché non serve)."""
        self.connection = get_connection()
        self.channel = get_channel(self.connection)

        if not self.connection or not self.channel:
            print("Errore di connessione a RabbitMQ.")
        else:
            print("Connessione a RabbitMQ stabilita correttamente.")

    def process_job(self, ch, method, properties, body):
        """Esegue la coroutine process_job nel loop asyncio"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.process_job_exec(ch, method, properties, body))

    async def process_job_exec(self, ch, method, properties, body):
        data = json.loads(body.decode())
        print(f"Processing job: {data}")        
        model_id = data.get("model_id")

        if not model_id:
            print("Errore: model_id mancante nel messaggio")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        
        # Passo 1: Leggere il documento da MongoDB
        model = model_service.get_model_by_id(model_id)

        if not model:
            print(f"Errore: Nessun documento trovato per model_id {model_id}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        
        video_uri = model.video_uri
        if not video_uri:
            print(f"Errore: Nessun video_uri trovato per model_id {model_id}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return


        try:
            # Passo 2: Creazione delle cartelle di lavoro e download del video
            model_dir = os.path.join(WORKING_DIR, f"{model_id}")
            os.makedirs(model_dir, exist_ok=True)

            model_service.update_model_status(model_id, {"status": "VIDEO_PROCESSING"})

            thumbnail_s3_key = self.process_video(model_id,model.video_uri,model_dir)
            
            model_service.update_model_status(model_id, {"status": "POINT_CLOUD_RECONSTRUCTION","thumbnail_s3_key": thumbnail_s3_key})

            self.build_point_cloud(model_dir)

            model_service.update_model_status(model_id, {"status": "MODEL_TRAINING"})

            output_s3_key = self.train_model(model_id,model_dir)
            
            # 🔹 Sempre aggiorniamo lo stato con entrambi i percorsi
            model_service.update_model_status(model_id, {"status": "COMPLETED","output_s3_key": output_s3_key})
            
        finally:
                # Questa parte verrà sempre eseguita, anche se c'è stata un'eccezione
                ch.basic_ack(delivery_tag=method.delivery_tag)
                print("Job completato o fallito, conferma del messaggio alla coda.")


    def process_video(self,model_id,video_uri,model_dir):
        video_file_path = os.path.join(model_dir, "video.mp4")
        repository_service.download(video_uri, video_file_path)

        # Estrarre fotogrammi dal video
        frames_output_folder = os.path.join(model_dir, 'input')
        thumbnail_path = self.extract_frames(video_file_path, frames_output_folder)

        thumbnail_s3_key = None
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                thumbnail_s3_key = f"models/{model_id}/thumbnail.jpg"
                repository_service.upload(thumbnail_path, thumbnail_s3_key)
                print(f"✅ Thumbnail caricata su S3: {thumbnail_s3_key}")
            except Exception as e:
                print(f"❌ Errore durante l'upload della thumbnail su S3: {e}")
                thumbnail_s3_key = None  # Se l'upload fallisce, non salviamo l'URL errato
            return thumbnail_s3_key

      
    def build_point_cloud(self,model_dir):
        # Chiamata API a Gaussian Splatting
        convert_request = {"input_dir": model_dir}
        response = requests.post(GAUSSIAN_SPLATTING_API_URL + "/convert", json=convert_request)
        if response.status_code != 200:
            print(f"Errore durante la conversione: {response.text}")
            return
        
    def train_model(self,model_id,model_dir):
        # Chiamata API per il training del modello
        train_output_folder = os.path.join(model_dir, 'output')
        train_request = {"input_dir": model_dir, "output_dir": train_output_folder}
        response = requests.post(GAUSSIAN_SPLATTING_API_URL + "/train", json=train_request)
        if response.status_code != 200:
            print(f"Errore durante il training: {response.text}")
            return

        # Zippa la cartella output
        zip_filename = os.path.join(model_dir, "3d_model.zip")
        shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', train_output_folder)

        # Carica lo ZIP su S3
        s3_key = f"models/{model_id}/3d_model.zip"
        try:
            repository_service.upload(zip_filename, s3_key)
        except Exception as e:
            print(f"Errore durante l'upload su S3: {e}")
            # Se l'upload fallisce, metti lo stato in "FAILED" o altro per segnalarlo
            model_service.update_model_status(model_id, {"status": "FAILED"})
            return
        return s3_key


    # Consuma i messaggi dalla coda
    def consume_jobs(self):
        
        # Inizia a consumare dalla coda 'job_queue'
        self.channel.basic_consume(queue='3dgs', on_message_callback=self.process_job)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()

        
    # Funzione per estrarre 120 immagini dal video
    def extract_frames(self, video_path, output_folder,threshold=0.5):  # Aumenta total_frames
        """Estrae fotogrammi ogni volta che c'è un cambiamento significativo nella scena"""

        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video {video_path}")
            return None  # Se il video non si apre, ritorna None

        previous_frame = None
        extracted_count = 0
        first_frame_path = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if previous_frame is None:
                previous_frame = frame
                continue

            # Calcola la differenza tra il fotogramma corrente e quello precedente
            diff = cv2.absdiff(previous_frame, frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

            # Calcola la percentuale di pixel che sono cambiati
            change_percentage = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])

            if change_percentage > threshold:
                image_filename = os.path.join(output_folder, f"frame_{extracted_count + 1}.jpg")
                cv2.imwrite(image_filename, frame)
                extracted_count += 1

                if first_frame_path is None:
                    first_frame_path = image_filename  # Salva il primo frame estratto

            previous_frame = frame  # Imposta il fotogramma corrente come fotogramma precedente

        cap.release()

        if first_frame_path:
            print(f"✅ Primo frame salvato: {first_frame_path}")
        else:
            print("❌ Nessun frame salvato.")

        return first_frame_path  # Restituisce il primo frame estratto



    def handle_exit(self, signum, frame):
        """Gestisce la chiusura dell'applicazione"""
        print("\n🛑 Closing application...")
        close_connection(self.connection)
        sys.exit(0)
