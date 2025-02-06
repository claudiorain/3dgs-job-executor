# Funzione per ottenere il client del database
import pika
import os

def get_connection():
    # Usa il valore di ambiente RABBITMQ_URI o una URL di defaultamqp://<user>:<password>@<hostname>:<port>
    rabbitmq_uri = os.getenv('RABBITMQ_URI')
    print('URI: ' + rabbitmq_uri)
    connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_uri))
    return connection

def get_channel(connection):
    channel = connection.channel()
    channel.queue_declare(queue='3dgs', durable=True)  # Assicuriamoci che la coda '3dgs' esista
    return channel

def close_connection(connection):
     if connection and connection.is_open:
        connection.close()
        print("RabbitMQ connection closed.")
