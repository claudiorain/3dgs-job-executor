from uuid import UUID
from config.db import get_database
from models.model import ModelResponse
from datetime import datetime
from pymongo import ReturnDocument

class ModelService:

    def __init__(self):
        """Inizializza la connessione a MongoDB."""
        self.db = get_database()

    def get_model_by_id(self, model_id: UUID) -> ModelResponse:
        """
        Recupera un modello dal database usando l'ID.
        """
        model = self.db['models'].find_one({"_id": str(model_id)})
        
        if model is None:
            return None

        # Mappa tutti i campi dalla nuova struttura
        return ModelResponse(
            _id=model['_id'],
            video_s3_key=model.get('video_s3_key'),
            model_s3_key=model.get('model_s3_key',None),
            thumbnail_s3_key=model.get('thumbnail_s3_key'),
            thumbnail_url=None,  # Viene generato dal frontend/API layer
            zip_model_url=None,  # Viene generato dal frontend/API layer
            parent_model_id=model.get('parent_model_id'),
            
            title=model['title'],
            description=model.get('description'),
            
            training_config=model.get('training_config'),
            phases=model.get('phases', {}),
            overall_status=model.get('overall_status', 'PENDING'),
            current_phase=model.get('current_phase'),
            results=model.get('results'),            
            created_at=model['created_at'],
            updated_at=model.get('updated_at')
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
            return_document=ReturnDocument.AFTER
        )

        return result
    
    def update_phase(self, model_id: UUID, phase: str, metadata: dict):
        """
        Aggiorna il metadata di una fase senza modificarne lo stato.

        :param model_id: UUID del modello
        :param phase: Nome della fase (es: "training", "colmap", ecc.)
        :param metadata: Dizionario di dati da salvare sotto phases.<phase>
        """
        update_fields = {
            "updated_at": datetime.utcnow()
        }

        for key, value in metadata.items():
            update_fields[f"phases.{phase}.{key}"] = value

        return self.update_model_status(model_id, update_fields)

    def start_phase(self, model_id: UUID, phase: str):
        """
        Helper per avviare una fase: aggiorna started_at e status RUNNING
        
        :param model_id: UUID del modello
        :param phase: Nome della fase (es: "frame_extraction", "colmap", "training", "metrics")
        """
        current_time = datetime.utcnow()
        
        update_fields = {
            "overall_status": "RUNNING",
            "current_phase": phase,
            f"phases.{phase}.status": "RUNNING", 
            f"phases.{phase}.started_at": current_time,
            "updated_at": current_time
        }
        
        return self.update_model_status(model_id, update_fields)
    
    
    def complete_phase(self, model_id: UUID, phase: str, overall_status: str = "RUNNING", **additional_data):
        """
        Helper per completare una fase: aggiorna completed_at e status COMPLETED
        
        :param model_id: UUID del modello
        :param phase: Nome della fase
        :param overall_status: Status generale del modello (default: "RUNNING")
        :param additional_data: Dati aggiuntivi da salvare (es: thumbnail_s3_key, metadata, etc.)
        """
        current_time = datetime.utcnow()
        
        update_fields = {
            "overall_status": overall_status,
            f"phases.{phase}.status": "COMPLETED",
            f"phases.{phase}.completed_at": current_time,
            "updated_at": current_time
        }
        
        # Aggiungi dati aggiuntivi alla fase
        for key, value in additional_data.items():
            if key.startswith('phases.'):
                # Se la chiave inizia già con phases., usala così com'è
                update_fields[key] = value
            else:
                # Altrimenti aggiungila alla fase specifica
                update_fields[f"phases.{phase}.{key}"] = value
        
        return self.update_model_status(model_id, update_fields)

    def fail_phase(self, model_id: UUID, phase: str, error_message: str):
        """
        Helper per fallire una fase: aggiorna error_message e status FAILED
        
        :param model_id: UUID del modello  
        :param phase: Nome della fase
        :param error_message: Messaggio di errore
        """
        current_time = datetime.utcnow()
        
        update_fields = {
            "overall_status": "FAILED",
            "current_phase": None,
            f"phases.{phase}.status": "FAILED",
            f"phases.{phase}.error_message": error_message,
            "updated_at": current_time
        }
        
        return self.update_model_status(model_id, update_fields)