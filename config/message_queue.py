# Funzione per ottenere il client del database
import pika
import os

hostname = os.getenv('RABBITMQ_HOSTNAME')
port = os.getenv('RABBITMQ_PORT')
user = os.getenv('RABBITMQ_USER')
password = os.getenv('RABBITMQ_PASS')

hostname = os.getenv('RABBITMQ_HOSTNAME')
port = os.getenv('RABBITMQ_PORT')
user = os.getenv('RABBITMQ_USER')
password = os.getenv('RABBITMQ_PASS')

def get_connection():
    # Usa il valore di ambiente RABBITMQ_URI o una URL di defaultamqp://<user>:<password>@<hostname>:<port>
    credentials = pika.PlainCredentials(user, password)
    parameters = pika.ConnectionParameters(hostname,
                                   port,
                                   '/',
                                   credentials,heartbeat=0,socket_timeout=None)

    connection = pika.BlockingConnection(parameters)
    return connection

def get_channel(connection):
    return connection.channel()

def close_connection(connection):
     if connection and connection.is_open:
        connection.close()
        print("RabbitMQ connection closed.")
