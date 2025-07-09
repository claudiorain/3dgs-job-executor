from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

class QualityLevel(str, Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    ULTRA = "ultra"

class AlgorithmMetadata(BaseModel):
    """Metadati dell'algoritmo"""
    description: str
    paper_reference: Optional[str] = None
    repository: Optional[str] = None
    supported_platforms: List[str] = Field(default_factory=list)
    min_gpu_memory_gb: int = 8
    notes: Optional[str] = None

class ResolutionThreshold(BaseModel):
    """Soglia per risoluzione automatica"""
    vram_threshold: float = Field(description="GB VRAM minima per questa risoluzione")
    resolution: int = Field(description="Valore resolution (-1 per auto, 1+ per fattori)")
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
    Modello generico per configurazione parametri training.
    Riutilizzabile per tutti gli algoritmi (GS, MCMC, Taming, etc.)
    """
    
    # === IDENTIFICAZIONE ===
    id: Optional[str] = Field(default=None, alias='_id')
    algorithm_name: str = Field(description="Nome univoco algoritmo")
    display_name: str = Field(description="Nome user-friendly")
    version: str = "1.0"
    active: bool = True
    
    # === METADATI ===
    metadata: AlgorithmMetadata
    
    # === PARAMETRI BASE (livello "standard") ===
    base_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Valori base per tutti i parametri dell'algoritmo"
    )
    
    # === MOLTIPLICATORI QUALITÀ ===
    quality_multipliers: Dict[QualityLevel, Dict[str, float]] = Field(
        default_factory=dict,
        description="Moltiplicatori per ogni livello di qualità"
    )
    
    # === OVERRIDE QUALITÀ (valori sostitutivi) ===
    quality_overrides: Dict[QualityLevel, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Valori che sostituiscono (non moltiplicano) per qualità"
    )
    
    # === CONFIGURAZIONE HARDWARE ===
    hardware_config: HardwareConfig
    
    # === PARAMETRI CALCOLATI POST-QUALITÀ ===
    post_calculation: Dict[str, CalculatedParam] = Field(
        default_factory=dict,
        description="Parametri calcolati dopo applicazione moltiplicatori"
    )
    
    # === LIMITI HARDWARE (stime, non parametri) ===
    hardware_limits: Dict[str, Any] = Field(
        default_factory=dict,
        description="Limiti e stime hardware (non vanno nel comando)"
    )
    
    # === VALIDAZIONI ===
    validation_rules: List[ValidationRule] = Field(
        default_factory=list,
        description="Regole di validazione parametri finali"
    )
    
    # === TIMESTAMP ===
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: str = "system"
    updated_by: str = "system"
    
    class Config:
        use_enum_values = True
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "algorithm_name": "gaussian_splatting_original",
                "display_name": "3D Gaussian Splatting (Original)",
                "version": "1.0",
                "active": True,
                "metadata": {
                    "description": "Algoritmo originale 3D Gaussian Splatting",
                    "min_gpu_memory_gb": 8
                },
                "base_params": {
                    "iterations": 30000,
                    "resolution": 1,
                    "opacity_lr": 0.05
                },
                "quality_multipliers": {
                    "draft": {
                        "iterations": 0.5,
                        "opacity_lr": 2.0
                    },
                    "standard": {
                        "iterations": 1.0,
                        "opacity_lr": 1.0
                    }
                }
            }
        }

class TrainingParamsResponse(BaseModel):
    """Response model per API"""
    id: str = Field(alias='_id')
    algorithm_name: str
    display_name: str
    version: str
    active: bool
    metadata: AlgorithmMetadata
    
    # Parametri semplificati per response
    available_quality_levels: List[QualityLevel]
    parameter_count: int
    hardware_requirements: Dict[str, Any]
    
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        allow_population_by_field_name = True

# === MODELLI PER GENERAZIONE PARAMETRI ===

class GenerateParamsRequest(BaseModel):
    """Richiesta per generare parametri"""
    algorithm_name: str
    quality_level: QualityLevel
    gpu_memory_gb: Optional[float] = None
    manual_overrides: Dict[str, Any] = Field(default_factory=dict)

class GeneratedParams(BaseModel):
    """Parametri generati pronti per l'uso"""
    algorithm_name: str
    quality_level: QualityLevel
    gpu_memory_gb: float
    
    # Parametri finali calcolati
    final_params: Dict[str, Any]
    
    # Metadati calcolo
    applied_quality_multipliers: Dict[str, float]
    applied_hardware_multipliers: Dict[str, float]
    applied_overrides: Dict[str, Any]
    calculated_params: Dict[str, Any]
    
    # Stime
    estimated_training_time_minutes: Optional[int] = None
    estimated_vram_usage_gb: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "algorithm_name": "gaussian_splatting_original",
                "quality_level": "high",
                "gpu_memory_gb": 16.0,
                "final_params": {
                    "iterations": 45000,
                    "resolution": 1,
                    "opacity_lr": 0.02,
                    "eval": True
                },
                "estimated_training_time_minutes": 90
            }
        }

# === HELPER FUNCTIONS ===

def create_gaussian_splatting_config() -> TrainingParamsConfig:
    """Helper per creare config Gaussian Splatting da zero"""
    return TrainingParamsConfig(
        algorithm_name="gaussian_splatting_original",
        display_name="3D Gaussian Splatting (Original)",
        metadata=AlgorithmMetadata(
            description="Algoritmo originale 3D Gaussian Splatting",
            min_gpu_memory_gb=8
        ),
        base_params={
            "iterations": 30000,
            "resolution": 1,
            "densify_grad_threshold": 0.0002,
            "opacity_lr": 0.05,
            "eval": True
        },
        quality_multipliers={
            QualityLevel.DRAFT: {"iterations": 0.5, "opacity_lr": 2.0},
            QualityLevel.STANDARD: {"iterations": 1.0, "opacity_lr": 1.0},
            QualityLevel.HIGH: {"iterations": 1.5, "opacity_lr": 0.4},
            QualityLevel.ULTRA: {"iterations": 2.0, "opacity_lr": 0.2}
        },
        hardware_config=HardwareConfig(),
        created_at=datetime.utcnow()
    )

def create_mcmc_config() -> TrainingParamsConfig:
    """Helper per creare config MCMC (esempio)"""
    return TrainingParamsConfig(
        algorithm_name="gaussian_splatting_mcmc",
        display_name="MCMC Gaussian Splatting",
        metadata=AlgorithmMetadata(
            description="Gaussian Splatting con MCMC sampling",
            min_gpu_memory_gb=8
        ),
        base_params={
            "max_gaussians": 2000000,
            "mcmc_steps": 100,
            "temperature": 0.1,
            "eval": True
        },
        quality_multipliers={
            QualityLevel.DRAFT: {"mcmc_steps": 0.5, "temperature": 2.0},
            QualityLevel.STANDARD: {"mcmc_steps": 1.0, "temperature": 1.0},
            QualityLevel.HIGH: {"mcmc_steps": 1.5, "temperature": 0.7},
            QualityLevel.ULTRA: {"mcmc_steps": 2.0, "temperature": 0.5}
        },
        hardware_config=HardwareConfig(),
        created_at=datetime.utcnow()
    )