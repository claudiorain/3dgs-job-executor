from config.message_queue import get_connection  # Assicurati che questa funzione restituisca il client del database
from config.message_queue import get_channel  # Assicurati che questa funzione restituisca il client del database
from services.model_service import ModelService

model_service = ModelService()

class QueueJobService:

    def __init__(self):
        """Inizializza la connessione a RabbitMQ (può rimanere None finché non serve)."""
        self.connection = get_connection()
        self.channel = get_channel(self.connection)

        if not self.connection or not self.channel:
            print("Errore di connessione a RabbitMQ.")
        else:
            print("Connessione a RabbitMQ stabilita correttamente.")


    def process_job(self,ch, method, properties, body):
        print(f"Processing job: {body.decode()}")
        # Aggiungi qui la logica per eseguire il job
        ch.basic_ack(delivery_tag=method.delivery_tag)

                # Scarica un file
        #s3_file_key = "video/mio-video.mp4"
        #local_file_path = "mio-video.mp4"

        #s3.download_file(s3_bucket, s3_file_key, local_file_path)

        #print(f"File scaricato: {local_file_path}")

    # Consuma i messaggi dalla coda
    def consume_jobs(self):
        
        # Inizia a consumare dalla coda 'job_queue'
        self.channel.basic_consume(queue='3dgs', on_message_callback=self.process_job)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()

        

    def handle_exit(self, signum, frame):
        """Gestisce la chiusura dell'applicazione"""
        print("\n🛑 Closing application...")
        close_connection(self.connection)
        sys.exit(0)
