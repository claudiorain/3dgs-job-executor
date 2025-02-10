# app/models/model.py
from pydantic import BaseModel, HttpUrl,Field
from uuid import UUID
from datetime import datetime

class ModelResponse(BaseModel):
    id: str = Field(alias='_id')
    filename: str
    model_name: str
    model_output_path: str
    status: str
    created_at: datetime
    updated_at: datetime

# Modello dati
class ModelCreateRequest(BaseModel):
    filename: str  
    model_name: str

    class Config:
        # Imposta come serializzare HttpUrl
        json_encoders = {
            HttpUrl: lambda v: str(v)  # Converti il tipo HttpUrl in stringa
        }

