#!/usr/bin/env python3
from config.message_queue import get_connection, get_channel, close_connection
from services.model_service import ModelService
from services.repository_service import RepositoryService
import json
import asyncio
import signal
import sys

# Importa la classe QueueJobService dal tuo file esistente
from services.queue_job_consumer_service import QueueJobService

queues = [
            'frame_extraction_queue',
            'point_cloud_queue',
            'model_training_queue',
            'upload_queue',
            'metrics_generation_queue'
        ]

class SequentialJobProcessor:
    def __init__(self):
        """Inizializza il processore di job sequenziale"""
        self.connection = get_connection()
        self.channel = get_channel(self.connection)
        self.model_service = ModelService()
        self.repository_service = RepositoryService()
        self.job_service = QueueJobService()
        self.running = True
        
        # Configura la gestione del segnale di interruzione
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
    
        # Crea le code se non esistono
        self.declare_queues()

    def declare_queues(self):
        """Dichiara le code se non esistono"""
        
        for queue_name in queues:
            # Dichiara la coda con il parametro durable=True, che significa che la coda sopravvive ai riavvii del server
            self.channel.queue_declare(queue=queue_name, durable=True)
            print(f"Queue '{queue_name}' dichiarata.")

    def start(self):
        """Avvia il processore sequenziale"""
        print(" [*] Starting sequential job processor")
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.process_queues_sequentially())
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            self.cleanup()
    
    async def process_queues_sequentially(self):
        """Processa le code in sequenza"""
       
        while self.running:
            job_processed = False
            
            for queue_name in queues:
                # Controlla se ci sono messaggi nella coda corrente
                method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=False)
                
                if method_frame:
                    print(f" [*] Processing job from {queue_name}")
                    # Processa il messaggio
                    await self.process_job(self.channel, method_frame, header_frame, body)
                    job_processed = True
                    # Dopo aver processato un job, ricomincia dalla prima coda
                    break
            
            # Se nessun job è stato processato in questo ciclo, attendi un po' prima di ricontrollare
            if not job_processed:
                await asyncio.sleep(5)  # Pausa di 5 secondi prima di ricontrollare le code
    
    def send_to_next_phase(self, model_id, next_queue, additional_data=None):
        """Invia il job alla fase successiva"""
        self.channel.basic_publish(
            exchange='',
            routing_key=next_queue,
            body=json.dumps({"model_id": model_id, "additional_data": additional_data})
        )
         
    async def process_job(self, ch, method, properties, body):
        """Processa un singolo job"""
        try:
            data = json.loads(body.decode())
            model_id = data.get("model_id")
            print(f"Elaborazione del job: {data}")
            
            if not model_id:
                print("Errore: model_id mancante nel messaggio")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            # Conferma messaggio dopo parsing
            ch.basic_ack(delivery_tag=method.delivery_tag)
    
            # FASE CONSUMER: Routing ed esecuzione handler
            success = False
            if method.routing_key == "frame_extraction_queue":
                success = await self.job_service.handle_frame_extraction(ch, method,model_id, data)
                next_queue = "point_cloud_queue" if success else None
                
            elif method.routing_key == "point_cloud_queue":
                success = await self.job_service.handle_point_cloud_building(ch, method,model_id, data)
                next_queue = "model_training_queue" if success else None

            elif method.routing_key == "model_training_queue":
                success = await self.job_service.handle_training(ch, method,model_id, data)
                next_queue = "upload_queue" if success else None
                
            elif method.routing_key == "upload_queue":
                success = await self.job_service.handle_model_upload(ch, method,model_id, data)
                next_queue = "metrics_generation_queue" if success else None
                
            elif method.routing_key == "metrics_generation_queue":
                success = await self.job_service.handle_metrics_generation(ch, method,model_id, data)
                next_queue = None  # Fine workflow
            
            # FASE PRODUCER: Gestione centralizzata della coda successiva
            if success and next_queue:
                self.send_to_next_phase(model_id, next_queue)
            
            
           
            print(f"Job completato: {method.routing_key} per model_id: {model_id}")
            
        except Exception as e:
            print(f"❌ Errore durante l'elaborazione del job: {e}")
            if 'model_id' in locals() and model_id:
                self.model_service.update_model_status(model_id, {"status": "ERROR", "error_message": str(e)})
            # Conferma comunque il messaggio per evitare di bloccarsi su messaggi problematici
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print("Job fallito, conferma del messaggio alla coda.")
    
    def handle_exit(self, signum, frame):
        """Gestisce la chiusura dell'applicazione"""
        print("\n🛑 Closing application...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Pulisce le risorse"""
        try:
            close_connection(self.connection)
            print("Connection closed")
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    processor = SequentialJobProcessor()
    processor.start()