# app/models/model.py
from pydantic import BaseModel, HttpUrl,Field
from uuid import UUID
from datetime import datetime
from typing import Optional


class ModelResponse(BaseModel):
    id: str = Field(alias='_id')
    video_uri: str
    thumbnail_s3_key: Optional[str] = None
    thumbnail_url: Optional[str] = None
    title: str
    output_s3_key: Optional[str] = None
    output_url: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None