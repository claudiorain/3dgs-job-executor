# app/models/model.py
from pydantic import BaseModel, HttpUrl,Field
from uuid import UUID
from datetime import datetime
from typing import Optional,Dict


class ModelResponse(BaseModel):
    id: str = Field(alias='_id')
    video_s3_key: str
    thumbnail_s3_key: Optional[str] = None
    thumbnail_url: Optional[str] = None
    title: str
    description: str
    engine: str
    output_s3_key: Optional[str] = None
    output_url: Optional[str] = None
    status: str
    results: Optional[Dict[str, float]] = None  # Aggiungi il campo results, che è opzionale
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None