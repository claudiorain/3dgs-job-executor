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

    def get_model_by_id(self, model_id: UUID) -> ModelResponse:
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
            video_uri=model['video_uri'],
            title=model['title'],
            output_path=model['output_path'],
            status=model['status'],
            created_at=model['created_at'],
            updated_at=model['updated_at']
        )
    
    def update_model_status(self, model_id: UUID, update_fields: dict):
        """
        Aggiorna lo stato, l'output path e il timestamp `updated_at` di un modello nel database.

        :param model_id: UUID del modello
        :param update_fields: Dizionario contenente i campi da aggiornare
        :return: Il documento aggiornato o None se non esiste
        """
         # Aggiungiamo sempre anche l'updated_at per tracciare la data di modifica
        update_fields["updated_at"] = datetime.utcnow()

        result = self.db['models'].find_one_and_update(
            {"_id": str(model_id)},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER  # Restituisce il documento aggiornato
        )

        return result  # Restituisce il documento aggiornato o None se non trovato