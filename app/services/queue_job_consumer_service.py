import pika
import json
import time
import os
from app.config.message_queue import get_connection  # Assicurati che questa funzione restituisca il client del database
from app.config.message_queue import get_channel  # Assicurati che questa funzione restituisca il client del database

class QueueJobService:

    def __init__(self):
        """Inizializza la connessione a RabbitMQ (può rimanere None finché non serve)."""
        self.connection = get_connection()
        self.channel = get_channel(self.connection)

    def process_job(ch, method, properties, body):
        print(f"Processing job: {body.decode()}")
        # Aggiungi qui la logica per eseguire il job
        ch.basic_ack(delivery_tag=method.delivery_tag)

    # Consuma i messaggi dalla coda
    def consume_jobs():
        channel = connect_to_rabbitmq()
        
        # Assicurati che la coda esista
        channel.queue_declare(queue='3dgs')

        # Inizia a consumare dalla coda 'job_queue'
        channel.basic_consume(queue='3dgs', on_message_callback=process_job)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        channel.start_consuming()

        

    def handle_exit(self, signum, frame):
        """Gestisce la chiusura dell'applicazione"""
        print("\n🛑 Closing application...")
        close_connection(self.connection)
        sys.exit(0)
