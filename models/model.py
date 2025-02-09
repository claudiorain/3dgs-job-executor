# app/models/model.py
from pydantic import BaseModel, HttpUrl,Field
from uuid import UUID
from datetime import datetime

class ModelResponse(BaseModel):
    id: str = Field(alias='_id')
    video_url: str
    model_name: str
    model_folder_url: str
    status: str
    created_at: datetime
    updated_at: datetime

# Modello dati
class ModelCreateRequest(BaseModel):
    video_url: str  
    model_name: str

    class Config:
        # Imposta come serializzare HttpUrl
        json_encoders = {
            HttpUrl: lambda v: str(v)  # Converti il tipo HttpUrl in stringa
        }

