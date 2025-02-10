from uuid import UUID
from config.db import get_database  # Assicurati che questa funzione restituisca il client del database
from models.model import ModelResponse  # Assumendo che il tuo modello sia in models.py
from datetime import datetime
from uuid import UUID
from pymongo import ReturnDocument

# Configurazione MongoDB

# Esempio di connessione al DB
class ModelService:

    def __init__(self):
        """Inizializza la connessione a MongoDB."""
        self.db = get_database()  # Ottieni il database con il client asincrono

    async def get_model_by_id(self, model_id: UUID) -> ModelResponse:
        """
        Recupera un modello dal database usando l'ID.
        """
        # Supponiamo che tu abbia una collezione 'models' nel tuo database MongoDB
        model = self.db['models'].find_one({"_id": str(model_id)})
        
        # Se il modello non esiste
        if model is None:
            return None

        # Restituisci un oggetto del tipo ModelResponse
        return ModelResponse(
            _id=model['_id'],
            filename=model['filename'],
            model_name=model['model_name'],
            model_output_path=model['model_output_path'],
            status=model['status'],
            created_at=model['created_at'],
            updated_at=model['updated_at']
        )
    
    async def update_model_status(self, model_id: UUID, model_output_path: str,status: str):
        """
        Aggiorna lo stato, l'output path e il timestamp `updated_at` di un modello nel database.

        :param model_id: UUID del modello
        :param model_output_path: path dello zip cioè il training output
        :param status: Nuovo stato (es. 'COMPLETED')
        :return: Il documento aggiornato o None se non esiste
        """
        result = await self.db['models'].find_one_and_update(
            {"_id": str(model_id)},
            {"$set": {"status": status,"model_output_path": model_output_path, "updated_at": datetime.utcnow()}},
            return_document=ReturnDocument.AFTER  # Restituisce il documento aggiornato
        )

        return result  # Restituisce il documento aggiornato o None se non trovato