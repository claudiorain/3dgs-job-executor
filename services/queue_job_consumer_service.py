from config.message_queue import get_connection  # Assicurati che questa funzione restituisca il client del database
from config.message_queue import get_channel  # Assicurati che questa funzione restituisca il client del database
from config.message_queue import close_connection  # Assicurati che questa funzione restituisca il client del database
from services.model_service import ModelService
from services.repository_service import RepositoryService
import os
import json
import asyncio
import requests
import cv2

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

    async def process_job_exec(self,ch, method, properties, body):
        data = json.loads(body.decode())
        print(f"Processing job: {data}")        
        # Prendo l'id del documento (Mongo)
        model_id = data.get("model_id")

        if not model_id:
            print("Errore: model_id mancante nel messaggio")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        
        # Passo 1: Leggere il documento da MongoDB
        model = await model_service.get_model_by_id(model_id)

        if not model:
            print(f"Errore: Nessun documento trovato per model_id {model_id}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        
        video_url = model.video_url
        if not video_url:
            print(f"Errore: Nessun video_url trovato per model_id {model_id}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        
        print("WORKING DIR: " + WORKING_DIR)

        video_file_path = os.path.join(WORKING_DIR, f"{model_id}.mp4")

        repository_service.download(video_url,video_file_path)
        
        # Estraggo tot frames nella cartella input sotto WORKING_DIR
        output_folder = os.path.join(WORKING_DIR, 'input')  # La cartella dove salvare le immagini estratte

        self.extract_frames(video_file_path, output_folder)
        
        # Chiamata API a Gaussian Splatting
        payload = {"target_dir": WORKING_DIR}
        response = requests.post(GAUSSIAN_SPLATTING_API_URL + "/convert", json=payload)
        #os.system(f"docker exec gaussian-splatting-image python convert.py s {WORKING_DIR}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    # Consuma i messaggi dalla coda
    def consume_jobs(self):
        
        # Inizia a consumare dalla coda 'job_queue'
        self.channel.basic_consume(queue='3dgs', on_message_callback=self.process_job)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()

        
    # Funzione per estrarre 120 immagini dal video
    def extract_frames(self,video_path, output_folder, total_frames=120):
        """Estrae un numero specifico di fotogrammi da un video e li salva in una directory."""

   
        # Ora crea la cartella in sicurezza
        os.makedirs(output_folder, exist_ok=True)

        # Apre il video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video {video_path}")
            return
        
        # Ottieni il numero totale di fotogrammi nel video
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcola l'intervallo di frame da estrarre
        frame_interval = total_video_frames // total_frames
        
        # Estrai i fotogrammi
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break  # Uscire se non ci sono più fotogrammi
            
            # Se il frame è uno dei frame desiderati, lo salviamo
            if frame_count % frame_interval == 0:
                image_filename = os.path.join(output_folder, f"frame_{extracted_count + 1}.jpg")
                cv2.imwrite(image_filename, frame)  # Salva l'immagine
                extracted_count += 1

            frame_count += 1
            
            if extracted_count >= total_frames:
                break  # Se abbiamo estratto il numero di fotogrammi desiderato, interrompiamo

        cap.release()
        print(f"Estrazione completata. {extracted_count} fotogrammi salvati in {output_folder}")

    def handle_exit(self, signum, frame):
        """Gestisce la chiusura dell'applicazione"""
        print("\n🛑 Closing application...")
        close_connection(self.connection)
        sys.exit(0)
