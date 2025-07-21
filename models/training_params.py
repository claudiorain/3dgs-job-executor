from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

class QualityLevel(str, Enum):
    """Livelli di qualità - CORRETTI per coerenza con MongoDB"""
    FAST = "fast"
    BALANCED = "balanced"  # ✅ Corretto: ora combacia con MongoDB
    QUALITY = "quality"

class AlgorithmMetadata(BaseModel):
    """Metadati dell'algoritmo"""
    description: str
    paper_reference: Optional[str] = None
    repository: Optional[str] = None
    supported_platforms: List[str] = Field(default_factory=list)
    min_gpu_memory_gb: int = 8
    notes: Optional[str] = None

class ResolutionThreshold(BaseModel):
    """Soglia per risoluzione automatica - AGGIORNATO per resolution_scale_factor"""
    vram_threshold: float = Field(description="GB VRAM minima per questa risoluzione")
    target_width: int = Field(description="Width da utilizzare")
    target_height: int = Field(description="Height da utilizzare")

    description: Optional[str] = None

class ScalingFormula(BaseModel):
    """Formula per scaling hardware"""
    formula: str = Field(description="Formula matematica come stringa")
    description: str
    min: Optional[float] = None
    max: Optional[float] = None

class HardwareConfig(BaseModel):
    """Configurazione adattamento hardware"""
    baseline_vram_gb: float = 24
    min_vram_gb: float = 8
    resolution_thresholds: List[ResolutionThreshold] = Field(default_factory=list)
    scaling_formulas: Dict[str, ScalingFormula] = Field(default_factory=dict)

class CalculatedParam(BaseModel):
    """Parametro calcolato automaticamente"""
    formula: str = Field(description="Formula per calcolare il parametro")
    description: str

class ValidationRule(BaseModel):
    """Regola di validazione parametri"""
    rule: str = Field(description="Regola come stringa (es: 'param1 < param2')")
    message: str = Field(description="Messaggio errore se validazione fallisce")

class TrainingParamsConfig(BaseModel):
    """
    Modello AGGIORNATO per configurazione parametri training.
    Con separazione pulita tra training_params e preprocessing_params.
    """
    
    # === IDENTIFICAZIONE ===
    id: Optional[str] = Field(default=None, alias='_id')
    algorithm_name: str = Field(description="Nome univoco algoritmo")
    display_name: str = Field(description="Nome user-friendly")
    version: str = "1.0"
    active: bool = True
    
    # === METADATI ===
    metadata: AlgorithmMetadata
    
    # === PARAMETRI BASE (livello "balanced") ===
    base_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Valori base per tutti i parametri di training"
    )
    
    # === MOLTIPLICATORI QUALITÀ (solo training params) ===
    quality_multipliers: Dict[QualityLevel, Dict[str, float]] = Field(
        default_factory=dict,
        description="Moltiplicatori per ogni livello di qualità (solo training params)"
    )
    
    # 🆕 === PREPROCESSING PARAMS (sostituisce quality_overrides) ===
    preprocessing_params: Dict[QualityLevel, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Parametri di preprocessing per ogni livello di qualità (resolution_scale_factor, target_frames, etc.)"
    )
    
    # === CONFIGURAZIONE HARDWARE ===
    hardware_config: HardwareConfig
    
    # === PARAMETRI CALCOLATI POST-QUALITÀ ===
    post_calculation: Dict[str, CalculatedParam] = Field(
        default_factory=dict,
        description="Parametri calcolati dopo applicazione moltiplicatori (solo training params)"
    )
    
    # === LIMITI HARDWARE (stime, non parametri) ===
    hardware_limits: Dict[str, Any] = Field(
        default_factory=dict,
        description="Limiti e stime hardware (non vanno nel comando)"
    )
    
    # === VALIDAZIONI ===
    validation_rules: List[ValidationRule] = Field(
        default_factory=list,
        description="Regole di validazione parametri finali (solo training params)"
    )
    
    # === TIMESTAMP ===
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: str = "system"
    updated_by: str = "system"
    
    class Config:
        use_enum_values = True
        allow_population_by_field_name = True

class GeneratedParams(BaseModel):
    """
    Parametri generati pronti per l'uso - AGGIORNATO con separazione
    """
    algorithm_name: str
    quality_level: QualityLevel
    gpu_memory_gb: float
    
    # 🎯 Parametri finali di training - QUESTO È L'OUTPUT PRINCIPALE per il training
    final_params: Dict[str, Any] = Field(
        description="Parametri di training pronti per l'uso"
    )
    
    # 🆕 Parametri di preprocessing - NUOVO OUTPUT per frame extraction, etc.
    preprocessing_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parametri per preprocessing (target_width,target_height, target_frames, etc.)"
    )
    
    # Metadati essenziali
    applied_multipliers: Dict[str, float] = Field(
        default_factory=dict,
        description="Moltiplicatori hardware applicati ai training params"
    )
    applied_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Override manuali applicati ai training params"
    )
    
    # Stime opzionali
    estimated_training_time_minutes: Optional[int] = None
    estimated_vram_usage_gb: Optional[float] = None