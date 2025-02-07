from uuid import UUID
from config.db import get_database  # Assicurati che questa funzione restituisca il client del database
from models.model import ModelResponse  # Assumendo che il tuo modello sia in models.py

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
            video_url=model['video_url'],
            model_name=model['model_name'],
            model_folder_url=model['model_folder_url'],
            status=model['status'],
            created_at=model['created_at'],
            updated_at=model['updated_at']
        )
