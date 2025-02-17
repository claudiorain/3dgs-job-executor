# app/models/model.py
from pydantic import BaseModel, HttpUrl,Field
from uuid import UUID
from datetime import datetime

class ModelResponse(BaseModel):
    id: str = Field(alias='_id')
    video_uri: str
    thumbnail_s3_key: str
    thumbnail_url: str
    title: str
    output_s3_key: str
    output_url: str
    status: str
    created_at: datetime
    updated_at: datetime