# app/models/model.py
from pydantic import BaseModel,Field
from datetime import datetime
from typing import Optional,Dict
from enum import Enum


class PhaseStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class Phase(Enum):
    FRAME_EXTRACTION = "frame_extraction"
    POINT_CLOUD_BUILDING = "point_cloud_building"
    TRAINING = "training"
    UPLOAD = "upload"
    METRICS = "metrics_evaluation"

class Engine(Enum):
    INRIA = "INRIA"
    MCMC = "MCMC" 
    TAMING = "TAMING"

# Mappa fase -> coda
PHASE_TO_QUEUE = {
    Phase.FRAME_EXTRACTION: "frame_extraction_queue",
    Phase.POINT_CLOUD_BUILDING: "point_cloud_queue",
    Phase.TRAINING: "model_training_queue",
    Phase.UPLOAD: "upload_queue",
    Phase.METRICS: "metrics_generation_queue"
}

# Mappa inversa coda -> fase
QUEUE_TO_PHASE = {v: k for k, v in PHASE_TO_QUEUE.items()}

class PhaseResult(BaseModel):  # ← Resta qui
    status: PhaseStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None
    
class ModelResponse(BaseModel):
    id: str = Field(alias='_id')
    video_s3_key: str
    model_s3_key: Optional[str] = None  
    model_s3_url: Optional[str] = None 
    thumbnail_s3_key: Optional[str] = None
    thumbnail_url: Optional[str] = None
    zip_model_url:Optional[str] = None
    parent_model_id: Optional[str] = None
    # Solo quello che serve davvero
    title: str
    description: Optional[str] = None
    training_config: Optional[Dict] = None
    # Status delle fasi (questo è tutto quello che serve)
    phases: Dict[str, PhaseResult] = {}
    current_phase: Optional[str] = None
    overall_status: PhaseStatus

    created_at: datetime
    updated_at: Optional[datetime] = None