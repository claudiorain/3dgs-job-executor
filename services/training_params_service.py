import subprocess
import logging
import math
from typing import Optional, Dict, Any, Tuple
from models.training_params import (
    TrainingParamsConfig, 
    GeneratedParams, 
    QualityLevel
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

class TrainingParamsService:
    """
    Service SEMPLIFICATO per gestire parametri di training da MongoDB.
    Focus su funzionalità essenziali, sicurezza e maintainability.
    """
    
    def __init__(self):
        self.db = get_database()
    
    # === METODI PRINCIPALI ===
    
    def generate_params(
        self, 
        engine: Engine, 
        quality_level: QualityLevel,
        gpu_memory_gb: Optional[float] = None,
        manual_overrides: Dict[str, Any] = None
    ) -> GeneratedParams:
        """
        Genera parametri finali pronti per il training.
        SEMPLIFICATO: meno metadati, più focus sui risultati.
        """
        
        print("🎯 === INIZIO generate_params ===")
        print(f"🎯 Engine: {engine}")
        print(f"🎯 Quality Level: {quality_level}")
        print(f"🎯 GPU Memory: {gpu_memory_gb}")
        print(f"🎯 Manual Overrides: {manual_overrides}")
        
        try:
            # 1. Carica configurazione
            print("📋 Step 1: Caricamento configurazione...")
            config = self.get_config_by_engine(engine)
            if not config:
                logger.error(f"❌ Configurazione non trovata per engine: {engine}")
                raise ValueError(f"Configurazione non trovata per engine: {engine}")
            print(f"✅ Configurazione caricata: {config.algorithm_name}")
            
            # 2. Auto-rileva GPU se necessario
            print("🔍 Step 2: Rilevamento GPU...")
            if gpu_memory_gb is None:
                gpu_memory_gb = self._detect_gpu_memory()
            print(f"✅ GPU Memory: {gpu_memory_gb} GB")
            
            # 3. Applica moltiplicatori qualità
            print("⚙️ Step 3: Applicazione quality transforms...")
            params = self._apply_quality_transforms(
                config.base_params.copy(), 
                config.quality_multipliers.get(quality_level, {}),
                config.quality_overrides.get(quality_level, {})
            )
            print(f"✅ Params dopo quality: {params}")
            
            # 4. Applica moltiplicatori hardware
            print("🔧 Step 4: Applicazione hardware scaling...")
            params, hw_multipliers = self._apply_hardware_scaling(
                params,
                config.hardware_config,
                gpu_memory_gb
            )
            print(f"✅ Params dopo hardware scaling: {params}")
            print(f"✅ Hardware multipliers: {hw_multipliers}")
            
            # 5. Applica override manuali
            print("🛠️ Step 5: Applicazione override manuali...")
            applied_overrides = {}
            if manual_overrides:
                print(f"🛠️ Applicando overrides: {manual_overrides}")
                params.update(manual_overrides)
                applied_overrides = manual_overrides.copy()
            else:
                print("🛠️ Nessun override manuale")
            print(f"✅ Params dopo overrides: {params}")
            
            # 6. Calcola parametri derivati (se presenti)
            print("📊 Step 6: Calcolo parametri derivati...")
            if config.post_calculation:
                print("📊 Post calculation presente, eseguendo...")
                calculated = self._calculate_derived_params(params, config.post_calculation)
                params.update(calculated)
                print(f"✅ Parametri calcolati: {calculated}")
            else:
                print("📊 Nessun post calculation configurato")
            
            # 7. Valida parametri finali
            print("✅ Step 7: Validazione parametri...")
            self._validate_params(params, config.validation_rules)
            print("✅ Validazione completata")
            
            # 8. Pulisci parametri (arrotonda interi, etc.)
            print("🧹 Step 8: Pulizia parametri...")
            params = self._clean_params(params)
            print(f"✅ Params finali puliti: {params}")
            
            # 9. Crea response semplificato
            print("📦 Step 9: Creazione response...")
            result = GeneratedParams(
                algorithm_name=config.algorithm_name,
                quality_level=quality_level,
                gpu_memory_gb=gpu_memory_gb,
                final_params=params,
                applied_multipliers=hw_multipliers,
                applied_overrides=applied_overrides,
                estimated_training_time_minutes=self._estimate_training_time(params),
                estimated_vram_usage_gb=self._estimate_vram_usage(params, gpu_memory_gb)
            )
            
            print("🎉 === FINE generate_params - SUCCESSO ===")
            print(f"🎉 Risultato finale: {result}")
            return result
            
        except Exception as e:
            logger.error(f"💥 === ERRORE in generate_params ===")
            logger.error(f"💥 Tipo errore: {type(e).__name__}")
            logger.error(f"💥 Messaggio: {str(e)}")
            logger.error(f"💥 ================================")
            raise
    
    def get_config_by_engine(self, engine: Engine) -> Optional[TrainingParamsConfig]:
        """Ottieni configurazione tramite Engine enum"""
        print(f"📋 get_config_by_engine chiamato per: {engine}")
        
        algorithm_name = ENGINE_TO_ALGORITHM.get(engine)
        if not algorithm_name:
            logger.error(f"❌ Engine non supportato: {engine}")
            raise ValueError(f"Engine non supportato: {engine}")
        
        print(f"📋 Algorithm name mappato: {algorithm_name}")
        return self.get_config_by_algorithm(algorithm_name)
    
    def get_config_by_algorithm(self, algorithm_name: str) -> Optional[TrainingParamsConfig]:
        """Ottieni configurazione tramite algorithm_name"""
        print(f"📋 get_config_by_algorithm chiamato per: {algorithm_name}")
        
        try:
            print("📋 Eseguendo query MongoDB...")
            doc = self.db['training_params'].find_one({
                "algorithm_name": algorithm_name,
                "active": True
            })
            
            if not doc:
                logger.error(f"❌ Nessun documento trovato per: {algorithm_name}")
                return None
            
            print(f"✅ Documento trovato, convertendo in TrainingParamsConfig...")
            # Converti ObjectId in stringa
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            
            config = TrainingParamsConfig(**doc)
            print(f"✅ Config creato: {config.algorithm_name} v{config.version}")
            return config
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento configurazione {algorithm_name}: {e}")
            return None
    
    # === METODI PRIVATI SEMPLIFICATI ===
    
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
                print(f"GPU VRAM rilevata: {vram_gb:.1f} GB")
                return vram_gb
            else:
                logger.warning("Impossibile rilevare VRAM GPU, uso default 12GB")
                return 12.0
                
        except Exception as e:
            logger.error(f"Errore rilevamento GPU: {e}, uso default 12GB")
            return 12.0
    
    def _apply_quality_transforms(
        self, 
        base_params: Dict[str, Any], 
        quality_mults: Dict[str, float],
        quality_overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Applica moltiplicatori e override qualità"""
        
        result = base_params.copy()
        
         # DEBUG: Log completo per capire cosa succede
        print(f"🔍 DEBUG _apply_quality_transforms:")
        print(f"  base_params: {base_params}")
        print(f"  quality_mults: {quality_mults}")
        print(f"  quality_overrides: {quality_overrides}")

        # Applica moltiplicatori
        for param_name, multiplier in quality_mults.items():
            if param_name in result and isinstance(result[param_name], (int, float)):
                result[param_name] = result[param_name] * multiplier
        
        # Applica override (sostituiscono, non moltiplicano)
        result.update(quality_overrides)
        print(f"  result finale: {result}")

        return result
    
    def _apply_hardware_scaling(
    self, 
    params: Dict[str, Any], 
    hw_config: Any,
    gpu_memory_gb: float,
    quality_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:

        result = params.copy()
        applied_multipliers = {}

        # DEBUG: Log iniziale completo
        print(f"🔍 DEBUG _apply_hardware_scaling INIZIATO:")
        print(f"  GPU Memory: {gpu_memory_gb} GB")
        print(f"  Baseline VRAM: {hw_config.baseline_vram_gb} GB")
        print(f"  Input params: {params}")
        print(f"  Quality overrides: {quality_overrides}")

        # Calcola vram_factor
        vram_factor = gpu_memory_gb / hw_config.baseline_vram_gb
        print(f"  🧮 VRAM Factor = {gpu_memory_gb} / {hw_config.baseline_vram_gb} = {vram_factor:.4f}")

        # DEBUG: Log formule disponibili
        print(f"  📝 Formule hardware disponibili:")
        for param_name, formula_config in hw_config.scaling_formulas.items():
            print(f"    - {param_name}: {formula_config.formula} (min={formula_config.min}, max={formula_config.max})")

        # Applica formule scaling per ogni parametro
        for param_name, formula_config in hw_config.scaling_formulas.items():
            print(f"  🔧 Processando parametro: {param_name}")
            
            if param_name in result:
                original_value = result[param_name]
                print(f"    ✅ Parametro trovato con valore: {original_value}")
                
                multiplier = self._evaluate_safe_formula(
                    formula=formula_config.formula,
                    vram_factor=vram_factor,
                    gpu_memory_gb=gpu_memory_gb,
                    baseline_vram_gb=hw_config.baseline_vram_gb,
                    min_val=formula_config.min,
                    max_val=formula_config.max
                )
                print(f"    📐 Moltiplicatore calcolato: {multiplier:.4f}")
                
                if isinstance(result[param_name], (int, float)):
                    new_value = result[param_name] * multiplier
                    result[param_name] = new_value
                    applied_multipliers[param_name] = multiplier
                    
                    print(f"    ✅ APPLICATO: {original_value} * {multiplier:.4f} = {new_value}")
                else:
                    logger.warning(f"    ⚠️ Parametro non numerico: {result[param_name]} (tipo: {type(result[param_name])})")
            else:
                logger.warning(f"    ❌ Parametro {param_name} NON TROVATO nei params")

        # Gestisci resolution con logica personalizzata
        if "resolution" in result:
            print(f"  🖼️ Gestendo resolution...")
            
            # Risoluzione indicata dal quality_override, se presente
            quality_resolution = None
            if quality_overrides and "resolution" in quality_overrides:
                quality_resolution = quality_overrides["resolution"]
                print(f"    Quality resolution da override: {quality_resolution}")
            else:
                quality_resolution = result["resolution"]
                print(f"    Quality resolution da result: {quality_resolution}")

            # Risoluzione massima permessa dalla VRAM
            hw_resolution = self._determine_auto_resolution(hw_config.resolution_thresholds, gpu_memory_gb)
            print(f"    HW resolution calcolata: {hw_resolution}")

            # Se la resolution da quality è <= di quella hw, mantieni quality_resolution, altrimenti sostituisci
            if hw_resolution <= quality_resolution :
                result["resolution"] = quality_resolution
                print(f"    ✅ Mantengo quality resolution: {quality_resolution}")
            else:
                result["resolution"] = hw_resolution
                print(f"    🔧 Sostituisco con HW resolution: {hw_resolution}")

        # DEBUG: Log finale
        print(f"  📋 RISULTATO FINALE:")
        print(f"    Parametri finali: {result}")
        print(f"    Moltiplicatori applicati: {applied_multipliers}")
        print(f"🏁 DEBUG _apply_hardware_scaling COMPLETATO")

        return result, applied_multipliers

    def _evaluate_safe_formula(
        self, 
        formula: str,
        vram_factor: float,
        gpu_memory_gb: float,
        baseline_vram_gb: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> float:
        """
        Valuta formule SICURE senza eval().
        Supporta solo pattern comuni per hardware scaling.
        """
        
        print(f"    🧮 _evaluate_safe_formula:")
        print(f"      Formula: '{formula}'")
        print(f"      VRAM factor: {vram_factor:.4f}")
        print(f"      Min/Max: {min_val}/{max_val}")
        
        try:
            # Pattern supportati (più sicuri di eval generico)
            
            # Pattern: max(min_val, max_val - (vram_factor * coefficient))
            if formula.startswith("max(") and "vram_factor" in formula:
                print(f"      🔍 Pattern riconosciuto: max() con vram_factor")
                result = self._parse_max_vram_formula(formula, vram_factor)
                print(f"      🧮 Risultato max formula: {result:.4f}")
            
            # Pattern: min(max_val, base + (vram_factor * coefficient))
            elif formula.startswith("min(") and "vram_factor" in formula:
                print(f"      🔍 Pattern riconosciuto: min() con vram_factor")
                result = self._parse_min_vram_formula(formula, vram_factor)
                print(f"      🧮 Risultato min formula: {result:.4f}")
            
            # Pattern: semplice moltiplicazione
            elif "*" in formula and "vram_factor" in formula:
                print(f"      🔍 Pattern riconosciuto: moltiplicazione semplice")
                result = self._parse_simple_multiplication(formula, vram_factor)
                print(f"      🧮 Risultato moltiplicazione: {result:.4f}")
            
            # Pattern: formula lineare
            elif "+" in formula or "-" in formula:
                print(f"      🔍 Pattern riconosciuto: formula lineare")
                result = self._parse_linear_formula(formula, vram_factor)
                print(f"      🧮 Risultato lineare: {result:.4f}")
            
            else:
                logger.warning(f"      ⚠️ Formula non supportata: {formula}, uso default 1.0")
                return 1.0
            
            # Applica limiti
            original_result = result
            if min_val is not None:
                result = max(min_val, result)
            if max_val is not None:
                result = min(max_val, result)
            
            if original_result != result:
                print(f"      🚧 Limiti applicati: {original_result:.4f} -> {result:.4f}")
            
            print(f"      ✅ Risultato finale: {result:.4f}")
            return float(result)
            
        except Exception as e:
            logger.error(f"      ❌ Errore valutazione formula '{formula}': {e}")
            return 1.0  # Fallback sicuro
    
    def _parse_max_vram_formula(self, formula: str, vram_factor: float) -> float:
        """Parse: max(min_val, max_val - (vram_factor * coeff))"""
        print(f"        🔧 Parsing MAX formula: {formula}")
        
        # Rimuovi "max(" e ")"
        content = formula[4:-1]
        parts = content.split(", ")
        
        min_val = float(parts[0])
        print(f"        📊 Min value: {min_val}")
        
        # parts[1] dovrebbe essere qualcosa come "0.5 + (vram_factor * 0.5)"
        expr = parts[1]
        print(f"        📝 Expression prima del replace: {expr}")
        
        # Sostituisci vram_factor con il valore numerico
        expr_with_value = expr.replace("vram_factor", str(vram_factor))
        print(f"        📝 Expression dopo replace: {expr_with_value}")
        
        # Parsing dell'espressione
        if " + (" in expr_with_value and " * " in expr_with_value and ")" in expr_with_value:
            # Formato: "base + (number * coefficient)"
            base_str = expr_with_value.split(" + (")[0].strip()
            mult_part = expr_with_value.split(" + (")[1].replace(")", "").strip()
            
            base = float(base_str)
            # mult_part è tipo "0.6663411458333334 * 0.5"
            mult_values = mult_part.split(" * ")
            number = float(mult_values[0])
            coeff = float(mult_values[1])
            
            calculated = base + (number * coeff)
            print(f"        🧮 Calcolo: {base} + ({number} * {coeff}) = {calculated}")
            
        elif " - (" in expr_with_value and " * " in expr_with_value and ")" in expr_with_value:
            # Formato: "base - (number * coefficient)"
            base_str = expr_with_value.split(" - (")[0].strip()
            mult_part = expr_with_value.split(" - (")[1].replace(")", "").strip()
            
            base = float(base_str)
            mult_values = mult_part.split(" * ")
            number = float(mult_values[0])
            coeff = float(mult_values[1])
            
            calculated = base - (number * coeff)
            print(f"        🧮 Calcolo: {base} - ({number} * {coeff}) = {calculated}")
            
        else:
            # Fallback: prova a valutare come numero semplice
            calculated = float(expr_with_value)
            print(f"        🧮 Calcolo semplice: {calculated}")
        
        result = max(min_val, calculated)
        print(f"        ✅ Risultato max({min_val}, {calculated}) = {result}")
        
        return result
    
    def _parse_min_vram_formula(self, formula: str, vram_factor: float) -> float:
        """Parse: min(max_val, base + (vram_factor * coeff))"""
        print(f"        🔧 Parsing MIN formula: {formula}")
        
        content = formula[4:-1]
        parts = content.split(", ")
        
        max_val = float(parts[0])
        print(f"        📊 Max value: {max_val}")
        
        expr = parts[1]
        print(f"        📝 Expression prima del replace: {expr}")
        
        expr_with_value = expr.replace("vram_factor", str(vram_factor))
        print(f"        📝 Expression dopo replace: {expr_with_value}")
        
        if " + (" in expr_with_value and " * " in expr_with_value and ")" in expr_with_value:
            base_str = expr_with_value.split(" + (")[0].strip()
            mult_part = expr_with_value.split(" + (")[1].replace(")", "").strip()
            
            base = float(base_str)
            mult_values = mult_part.split(" * ")
            number = float(mult_values[0])
            coeff = float(mult_values[1])
            
            calculated = base + (number * coeff)
            print(f"        🧮 Calcolo: {base} + ({number} * {coeff}) = {calculated}")
            
        elif " - (" in expr_with_value and " * " in expr_with_value and ")" in expr_with_value:
            base_str = expr_with_value.split(" - (")[0].strip()
            mult_part = expr_with_value.split(" - (")[1].replace(")", "").strip()
            
            base = float(base_str)
            mult_values = mult_part.split(" * ")
            number = float(mult_values[0])
            coeff = float(mult_values[1])
            
            calculated = base - (number * coeff)
            print(f"        🧮 Calcolo: {base} - ({number} * {coeff}) = {calculated}")
            
        else:
            calculated = float(expr_with_value)
            print(f"        🧮 Calcolo semplice: {calculated}")
        
        result = min(max_val, calculated)
        print(f"        ✅ Risultato min({max_val}, {calculated}) = {result}")
        
        return result
    
    def _parse_simple_multiplication(self, formula: str, vram_factor: float) -> float:
        """Parse: vram_factor * coeff"""
        print(f"        🔧 Parsing MULTIPLICATION formula: {formula}")
        
        expr = formula.replace("vram_factor", str(vram_factor))
        parts = expr.split(" * ")
        result = float(parts[0]) * float(parts[1])
        
        print(f"        🧮 Calcolo: {parts[0]} * {parts[1]} = {result}")
        
        return result

    def _parse_linear_formula(self, formula: str, vram_factor: float) -> float:
        """Parse: a + b * vram_factor o a - b * vram_factor"""
        print(f"        🔧 Parsing LINEAR formula: {formula}")
        
        expr = formula.replace("vram_factor", str(vram_factor))
        
        if " - " in expr:
            parts = expr.split(" - ")
            result = float(parts[0]) - float(parts[1])
            print(f"        🧮 Calcolo: {parts[0]} - {parts[1]} = {result}")
        elif " + " in expr:
            parts = expr.split(" + ")
            result = float(parts[0]) + float(parts[1])
            print(f"        🧮 Calcolo: {parts[0]} + {parts[1]} = {result}")
        else:
            result = float(expr)
            print(f"        🧮 Calcolo semplice: {result}")
        
        return result
    
    def _determine_auto_resolution(self, thresholds: list, gpu_memory_gb: float) -> int:
        """Determina risoluzione automatica basata su VRAM"""
        sorted_thresholds = sorted(thresholds, key=lambda x: x.vram_threshold, reverse=True)
        
        for threshold in sorted_thresholds:
            if round(gpu_memory_gb) >= threshold.vram_threshold:
                return threshold.resolution
        
        return 8  # Fallback conservativo
    
    def _calculate_derived_params(
        self, 
        params: Dict[str, Any], 
        post_calc_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcola parametri derivati SEMPLIFICATO.
        Supporta solo formule base senza eval().
        """
        
        calculated = {}
        
        for param_name, calc_config in post_calc_config.items():
            try:
                formula = calc_config.get("formula", "") if isinstance(calc_config, dict) else calc_config.formula
                
                if not formula:
                    continue
                
                # Supporta solo pattern comuni per post-calculation
                if "iterations" in formula and "*" in formula:
                    # Esempio: "iterations * 0.5"
                    if " * " in formula:
                        base_param = formula.split(" * ")[0].strip()
                        multiplier = float(formula.split(" * ")[1].strip())
                        if base_param in params:
                            calculated[param_name] = int(params[base_param] * multiplier)
                
                elif "round(" in formula and ")" in formula:
                    # Esempio: "round(iterations * 0.017)"
                    inner = formula[6:-1]  # Rimuovi "round(" e ")"
                    if " * " in inner:
                        base_param = inner.split(" * ")[0].strip()
                        multiplier = float(inner.split(" * ")[1].strip())
                        if base_param in params:
                            calculated[param_name] = round(params[base_param] * multiplier)
                
                print(f"✅ Calcolato {param_name} = {calculated.get(param_name)}")
                
            except Exception as e:
                logger.error(f"❌ Errore calcolo parametro {param_name}: {e}")
        
        return calculated
    
    def _validate_params(self, params: Dict[str, Any], validation_rules: list):
        """Valida parametri finali - SEMPLIFICATO"""
        
        for rule in validation_rules:
            try:
                rule_str = rule.rule if hasattr(rule, 'rule') else rule.get('rule', '')
                message = rule.message if hasattr(rule, 'message') else rule.get('message', 'Validazione fallita')
                
                # Validazioni hardcoded (più sicure di eval)
                if rule_str == "densify_until_iter < iterations":
                    if params.get("densify_until_iter", 0) >= params.get("iterations", 30000):
                        raise ValueError(message)
                
                elif rule_str == "densify_from_iter < densify_until_iter":
                    if params.get("densify_from_iter", 0) >= params.get("densify_until_iter", 15000):
                        raise ValueError(message)
                
                elif rule_str == "densify_until_iter <= iterations * 0.8":
                    max_allowed = params.get("iterations", 30000) * 0.8
                    if params.get("densify_until_iter", 0) > max_allowed:
                        params["densify_until_iter"] = int(max_allowed)  # Auto-fix
                
                elif rule_str == "iterations >= 1000":
                    if params.get("iterations", 30000) < 1000:
                        raise ValueError(message)
                
                elif rule_str == "sh_degree <= 3":
                    if params.get("sh_degree", 3) > 3:
                        raise ValueError(message)
                        
            except ValueError:
                raise  # Re-raise validation errors
            except Exception as e:
                logger.error(f"Errore validazione regola '{rule_str}': {e}")
    
    def _clean_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pulisce i parametri: arrotonda interi, formatta float"""
        
        # Parametri che devono essere interi
        int_params = {
            'iterations', 'cap_max', 'densification_interval', 'densify_until_iter',
            'densify_from_iter', 'opacity_reset_interval', 'position_lr_max_steps',
            'sh_degree', 'budget', 'ho_iteration', 'test_iterations', 'cams'
        }
        
        # Parametri che devono essere float con precisione limitata
        float_params = {
            'densify_grad_threshold', 'opacity_lr', 'scaling_lr', 'rotation_lr',
            'position_lr_init', 'position_lr_final', 'position_lr_delay_mult',
            'percent_dense', 'lambda_dssim', 'noise_lr', 'scale_reg', 'opacity_reg', 
            'feature_lr', 'shfeature_lr'
        }
        
        cleaned = params.copy()
        
        for param_name, value in cleaned.items():
            if param_name in int_params and isinstance(value, (int, float)):
                cleaned[param_name] = int(round(value))
            elif param_name in float_params and isinstance(value, (int, float)):
                cleaned[param_name] = round(float(value), 8)
        
        return cleaned
    
    def _estimate_training_time(self, params: Dict[str, Any]) -> int:
        """Stima tempo training in minuti"""
        iterations = params.get("iterations", 30000)
        return max(10, int(iterations / 500))
    
    def _estimate_vram_usage(self, params: Dict[str, Any], gpu_memory_gb: float) -> float:
        """Stima uso VRAM"""
        base_usage = gpu_memory_gb * 0.8
        resolution = params.get("resolution", 1)
        resolution_factor = 1.0 / max(1, resolution)
        return base_usage * resolution_factor
    
    # === METODI UTILITY ===
    
    def list_available_algorithms(self) -> list[str]:
        """Lista algoritmi disponibili"""
        try:
            docs = list(self.db['training_params'].find({"active": True}, {"algorithm_name": 1}))
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