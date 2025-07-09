# app/services/training_params_service.py
import subprocess
import logging
import math
import re
from typing import Optional, Dict, Any, Union, Tuple
from models.training_params import (
    TrainingParamsConfig, 
    GeneratedParams, 
    QualityLevel,
    ResolutionThreshold
)
from config.db import get_database
from models.model import Engine  # Il tuo enum esistente

logger = logging.getLogger(__name__)

# Mapping tra Engine e algorithm_name
ENGINE_TO_ALGORITHM = {
    Engine.INRIA: "gaussian_splatting_original",
    Engine.MCMC: "gaussian_splatting_mcmc", 
    Engine.TAMING: "taming_3dgs"
}

ALGORITHM_TO_ENGINE = {v: k for k, v in ENGINE_TO_ALGORITHM.items()}

class TrainingParamsService:
    """
    Service per gestire parametri di training da MongoDB.
    Supporta tutti gli algoritmi in modo generico.
    """
    
    def __init__(self):
        """Inizializza la connessione a MongoDB"""
        self.db = get_database()
    
    def _clean_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pulisce i parametri: arrotonda interi, formatta float"""
        
        # Parametri che devono essere interi
        int_params = {
            'iterations','cap_max', 'densification_interval', 'densify_until_iter',
            'densify_from_iter', 'opacity_reset_interval', 'position_lr_max_steps',
            'sh_degree', 'budget', 'ho_iteration', 'test_iterations',"cams"  # Aggiunto MCMC
        }
        
        # Parametri che devono essere float con precisione limitata
        float_params = {
            'densify_grad_threshold', 'opacity_lr', 'scaling_lr', 'rotation_lr',
            'position_lr_init', 'position_lr_final', 'position_lr_delay_mult',
            'percent_dense', 'lambda_dssim', 'noise_lr',
            'scale_reg', 'opacity_reg', 'feature_lr', 'shfeature_lr'  # Aggiunti Taming
        }
        
        cleaned = params.copy()
        
        for param_name, value in cleaned.items():
            if param_name in int_params and isinstance(value, (int, float)):
                cleaned[param_name] = int(round(value))
            elif param_name in float_params and isinstance(value, (int, float)):
                # Limita a 8 decimali per evitare float strani
                cleaned[param_name] = round(float(value), 8)
        
        return cleaned

    # === METODI PRINCIPALI ===
    
    def get_config_by_engine(self, engine: Engine) -> Optional[TrainingParamsConfig]:
        """Ottieni configurazione tramite Engine enum"""
        algorithm_name = ENGINE_TO_ALGORITHM.get(engine)
        if not algorithm_name:
            raise ValueError(f"Engine non supportato: {engine}")
        
        return self.get_config_by_algorithm(algorithm_name)
    
    def get_config_by_algorithm(self, algorithm_name: str) -> Optional[TrainingParamsConfig]:
        """Ottieni configurazione tramite algorithm_name"""
        try:
            doc = self.db['training_params'].find_one({
                "algorithm_name": algorithm_name,
                "active": True
            })
            
            if not doc:
                return None
            
            # ✅ Fix: Converti ObjectId in stringa
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            
            return TrainingParamsConfig(**doc)
            
        except Exception as e:
            logger.error(f"Errore caricamento configurazione {algorithm_name}: {e}")
            return None
    
    def generate_params(
        self, 
        engine: Engine, 
        quality_level: QualityLevel,
        gpu_memory_gb: Optional[float] = None,
        manual_overrides: Dict[str, Any] = None
    ) -> GeneratedParams:
        """
        Genera parametri finali pronti per il training.
        
        Args:
            engine: Engine di training
            quality_level: Livello di qualità
            gpu_memory_gb: VRAM GPU (auto-rilevata se None)
            manual_overrides: Override manuali parametri
        """
        
        # 1. Carica configurazione
        config = self.get_config_by_engine(engine)
        if not config:
            raise ValueError(f"Configurazione non trovata per engine: {engine}")
        
        # 2. Auto-rileva GPU se necessario
        if gpu_memory_gb is None:
            gpu_memory_gb = self._detect_gpu_memory()
        
        # 3. Applica moltiplicatori qualità
        params_with_quality = self._apply_quality_multipliers(
            config.base_params.copy(), 
            config.quality_multipliers.get(quality_level, {}),
            config.quality_overrides.get(quality_level, {})
        )
        
        # 4. Applica moltiplicatori hardware
        params_with_hardware, hw_multipliers = self._apply_hardware_multipliers(
            params_with_quality,
            config.hardware_config,
            gpu_memory_gb
        )
        
        # 5. Applica override manuali
        if manual_overrides:
            params_with_hardware.update(manual_overrides)
        
        # 6. Calcola parametri derivati
        calculated_params = self._calculate_derived_params(
            params_with_hardware,
            config.post_calculation
        )
        params_with_hardware.update(calculated_params)
        
        # 7. Valida parametri finali
        self._validate_params(params_with_hardware, config.validation_rules)
        
        # 8. Pulisci parametri prima di restituire
        params_with_hardware = self._clean_params(params_with_hardware)
        
        # 9. Crea response
        return GeneratedParams(
            algorithm_name=config.algorithm_name,
            quality_level=quality_level,
            gpu_memory_gb=gpu_memory_gb,
            final_params=params_with_hardware,
            applied_quality_multipliers=config.quality_multipliers.get(quality_level, {}),
            applied_hardware_multipliers=hw_multipliers,
            applied_overrides=manual_overrides or {},
            calculated_params=calculated_params,
            estimated_training_time_minutes=self._estimate_training_time(params_with_hardware),
            estimated_vram_usage_gb=self._estimate_vram_usage(params_with_hardware, gpu_memory_gb)
        )
    
    # === METODI PRIVATI ===
    
    def _detect_gpu_memory(self) -> float:
        """Auto-rileva VRAM GPU"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.total', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                vram_mb = int(result.stdout.strip())
                vram_gb = vram_mb / 1024
                logger.info(f"GPU VRAM rilevata: {vram_gb:.1f} GB")
                return vram_gb
            else:
                logger.warning("Impossibile rilevare VRAM GPU, uso default 12GB")
                return 12.0
                
        except Exception as e:
            logger.error(f"Errore rilevamento GPU: {e}, uso default 12GB")
            return 12.0
    
    def _apply_quality_multipliers(
        self, 
        base_params: Dict[str, Any], 
        quality_mults: Dict[str, float],
        quality_overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Applica moltiplicatori e override qualità"""
        
        result = base_params.copy()
        
        # Applica moltiplicatori
        for param_name, multiplier in quality_mults.items():
            if param_name in result and isinstance(result[param_name], (int, float)):
                result[param_name] = result[param_name] * multiplier
        
        # Applica override (sostituiscono, non moltiplicano)
        result.update(quality_overrides)
        
        return result
    
    def _apply_hardware_multipliers(
        self, 
        params: Dict[str, Any], 
        hw_config: Any,
        gpu_memory_gb: float
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Applica moltiplicatori hardware usando formula generica"""
        
        result = params.copy()
        applied_multipliers = {}
        
        # Calcola vram_factor
        vram_factor = gpu_memory_gb / hw_config.baseline_vram_gb
        
        # Variabili disponibili per le formule hardware
        hardware_variables = {
            "vram_factor": vram_factor,
            "gpu_memory_gb": gpu_memory_gb,
            "baseline_vram_gb": hw_config.baseline_vram_gb
        }
        
        # Applica formule scaling
        for param_name, formula_config in hw_config.scaling_formulas.items():
            if param_name in result:
                multiplier = self._evaluate_formula(
                    formula=formula_config.formula,
                    variables=hardware_variables,
                    min_val=formula_config.min,
                    max_val=formula_config.max,
                    fallback_value=1.0
                )
                
                if isinstance(result[param_name], (int, float)):
                    result[param_name] = result[param_name] * multiplier
                    applied_multipliers[param_name] = multiplier
        
        # Gestisci resolution automatica
        if "resolution" in result and result["resolution"] == -1:
            auto_resolution = self._determine_auto_resolution(
                hw_config.resolution_thresholds, 
                gpu_memory_gb
            )
            result["resolution"] = auto_resolution
        
        return result, applied_multipliers
    
    def _evaluate_formula(
        self, 
        formula: str, 
        variables: Dict[str, Union[int, float]] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        fallback_value: float = 1.0
    ) -> float:
        """
        Valuta formula matematica con sostituzione di variabili multiple.
        
        Args:
            formula: Formula come stringa (es: "max(500, iterations * 0.017)")
            variables: Dict con variabili da sostituire (es: {"iterations": 30000, "vram_factor": 1.5})
            min_val: Valore minimo del risultato
            max_val: Valore massimo del risultato  
            fallback_value: Valore di fallback in caso di errore
        """
        
        try:
            formula_eval = formula
            
            # Sostituisci tutte le variabili nella formula
            if variables:
                # Ordina per lunghezza decrescente per evitare sostituzioni parziali
                sorted_vars = sorted(variables.items(), key=lambda x: len(x[0]), reverse=True)
                
                for var_name, var_value in sorted_vars:
                    # Sostituisci solo parole complete usando word boundaries
                    pattern = r'\b' + re.escape(var_name) + r'\b'
                    formula_eval = re.sub(pattern, str(var_value), formula_eval)
            
            # Valuta formule supportate
            if formula_eval.startswith("max("):
                result = self._evaluate_max_function(formula_eval)
            elif formula_eval.startswith("min("):
                result = self._evaluate_min_function(formula_eval)
            elif formula_eval.startswith("round("):
                result = self._evaluate_round_function(formula_eval)
            elif formula_eval.startswith("int("):
                result = self._evaluate_int_function(formula_eval)
            else:
                # Formula matematica semplice
                result = self._evaluate_simple_expression(formula_eval)
            
            # Applica limiti
            if min_val is not None:
                result = max(min_val, result)
            if max_val is not None:
                result = min(max_val, result)
            
            return float(result)
            
        except Exception as e:
            logger.error(f"Errore valutazione formula '{formula}': {e}")
            return fallback_value

    def _evaluate_max_function(self, formula_eval: str) -> float:
        """Valuta funzione max(a, b, c, ...)"""
        content = formula_eval[4:-1]  # Rimuovi "max(" e ")"
        
        if ", " in content:
            parts = content.split(", ")
            values = [self._evaluate_simple_expression(part.strip()) for part in parts]
            return max(values)
        else:
            return self._evaluate_simple_expression(content)

    def _evaluate_min_function(self, formula_eval: str) -> float:
        """Valuta funzione min(a, b, c, ...)"""
        content = formula_eval[4:-1]  # Rimuovi "min(" e ")"
        
        if ", " in content:
            parts = content.split(", ")
            values = [self._evaluate_simple_expression(part.strip()) for part in parts]
            return min(values)
        else:
            return self._evaluate_simple_expression(content)

    def _evaluate_round_function(self, formula_eval: str) -> float:
        """Valuta funzione round(x) o round(x, decimals)"""
        content = formula_eval[6:-1]  # Rimuovi "round(" e ")"
        
        if ", " in content:
            parts = content.split(", ")
            value = self._evaluate_simple_expression(parts[0].strip())
            decimals = int(self._evaluate_simple_expression(parts[1].strip()))
            return round(value, decimals)
        else:
            value = self._evaluate_simple_expression(content)
            return round(value)

    def _evaluate_int_function(self, formula_eval: str) -> float:
        """Valuta funzione int(x)"""
        content = formula_eval[4:-1]  # Rimuovi "int(" e ")"
        value = self._evaluate_simple_expression(content)
        return float(int(value))

    def _evaluate_simple_expression(self, expr: str) -> float:
        """
        Valuta espressione matematica semplice usando eval sicuro.
        Solo numeri e operatori matematici di base.
        """
        
        # Verifica che l'espressione sia sicura
        if not self._is_safe_math_expression(expr):
            raise ValueError(f"Espressione non sicura: {expr}")
        
        # Namespace limitato per eval
        safe_dict = {
            "__builtins__": {},
            # Aggiungi solo quello che serve
        }
        
        return float(eval(expr, safe_dict, {}))

    def _is_safe_math_expression(self, expr: str) -> bool:
        """Verifica che l'espressione contenga solo caratteri matematici sicuri"""
        
        # Pattern per caratteri permessi: numeri, operatori, parentesi, punto decimale
        safe_pattern = re.compile(r'^[0-9+\-*/.() ]+$')
        
        if not safe_pattern.match(expr):
            return False
        
        # Verifica bilanciamento parentesi
        if expr.count('(') != expr.count(')'):
            return False
        
        return True
    
    def _determine_auto_resolution(
        self, 
        thresholds: list, 
        gpu_memory_gb: float
    ) -> int:
        """Determina risoluzione automatica basata su VRAM"""
        
        # Ordina thresholds per vram_threshold decrescente
        sorted_thresholds = sorted(thresholds, key=lambda x: x.vram_threshold, reverse=True)
        
        for threshold in sorted_thresholds:
            if gpu_memory_gb >= threshold.vram_threshold:
                return threshold.resolution
        
        # Fallback
        return -1
    
    def _calculate_derived_params(
        self, 
        params: Dict[str, Any], 
        post_calc_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcola parametri derivati usando l'evaluatore di formule generico.
        Ora supporta qualsiasi formula senza modifiche al codice!
        """
        
        calculated = {}
        
        for param_name, calc_config in post_calc_config.items():
            try:
                # Gestisci sia dict che oggetti CalculatedParam
                if hasattr(calc_config, 'formula'):
                    # È un oggetto CalculatedParam
                    formula = calc_config.formula
                    description = getattr(calc_config, 'description', "")
                else:
                    # È un dict
                    formula = calc_config.get("formula", "")
                    description = calc_config.get("description", "")
                
                if not formula:
                    logger.warning(f"Formula vuota per {param_name}")
                    continue
                
                logger.debug(f"Calcolando {param_name}: {formula}")
                
                # Valuta la formula usando l'evaluatore generico
                result = self._evaluate_formula(
                    formula=formula,
                    variables=params,  # Tutti i parametri sono disponibili come variabili
                    fallback_value=0.0
                )
                
                calculated[param_name] = result
                logger.info(f"✅ {param_name} = {result} (da: {formula})")
                
            except Exception as e:
                logger.error(f"❌ Errore calcolo parametro {param_name}: {e}")
                logger.debug(f"Tipo calc_config: {type(calc_config)}")
                logger.debug(f"Attributi calc_config: {dir(calc_config) if hasattr(calc_config, '__dict__') else 'N/A'}")
                # Non interrompere il processo, continua con altri parametri
        
        return calculated
    
    def _validate_params(self, params: Dict[str, Any], validation_rules: list):
        """Valida parametri finali"""
        
        for rule in validation_rules:
            try:
                rule_str = rule.rule
                
                # Validazioni supportate
                if rule_str == "densify_until_iter < iterations":
                    if params.get("densify_until_iter", 0) >= params.get("iterations", 30000):
                        raise ValueError(rule.message)
                elif rule_str == "densify_from_iter < densify_until_iter":
                    if params.get("densify_from_iter", 0) >= params.get("densify_until_iter", 15000):
                        raise ValueError(rule.message)
                elif rule_str == "densify_until_iter <= iterations * 0.75":
                    max_allowed = params.get("iterations", 30000) * 0.75
                    if params.get("densify_until_iter", 0) > max_allowed:
                        params["densify_until_iter"] = int(max_allowed)  # Auto-fix
                elif rule_str == "iterations >= 1000":
                    if params.get("iterations", 30000) < 1000:
                        raise ValueError(rule.message)
                        
            except ValueError:
                raise  # Re-raise validation errors
            except Exception as e:
                logger.error(f"Errore validazione regola '{rule.rule}': {e}")
    
    def _estimate_training_time(self, params: Dict[str, Any]) -> int:
        """Stima tempo training in minuti"""
        iterations = params.get("iterations", 30000)
        # Stima approssimativa: ~500 iter/minuto
        return max(10, int(iterations / 500))
    
    def _estimate_vram_usage(self, params: Dict[str, Any], gpu_memory_gb: float) -> float:
        """Stima uso VRAM"""
        # Stima conservativa: 70-90% della VRAM disponibile
        base_usage = gpu_memory_gb * 0.8
        
        # Adjust per resolution
        resolution = params.get("resolution", 1)
        if resolution == -1:
            resolution_factor = 0.8  # Auto resolution usa meno
        else:
            resolution_factor = 1.0 / resolution  # Fattori più alti = meno VRAM
        
        return base_usage * resolution_factor
    
    # === METODI UTILITY ===
    
    def list_available_algorithms(self) -> list[str]:
        """Lista algoritmi disponibili"""
        try:
            cursor = self.db['training_params'].find({"active": True}, {"algorithm_name": 1})
            docs = cursor.to_list(length=None)
            return [doc["algorithm_name"] for doc in docs]
        except Exception as e:
            logger.error(f"Errore caricamento algoritmi: {e}")
            return []
    
    def get_algorithm_info(self, engine: Engine) -> Optional[Dict[str, Any]]:
        """Ottieni info base algoritmo"""
        config = self.get_config_by_engine(engine)
        if not config:
            return None
        
        return {
            "algorithm_name": config.algorithm_name,
            "display_name": config.display_name,
            "version": config.version,
            "description": config.metadata.description,
            "min_gpu_memory_gb": config.metadata.min_gpu_memory_gb,
            "available_quality_levels": list(config.quality_multipliers.keys()),
            "parameter_count": len(config.base_params)
        }