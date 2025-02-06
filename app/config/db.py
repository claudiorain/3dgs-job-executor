from pymongo import MongoClient
import os

# Funzione per ottenere il client del database
def get_database():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://mongo:27017"))
    return client.get_database()  # Restituisce direttamente il database


