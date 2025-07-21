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
    Service per gestire parametri di training da MongoDB.
    Con separazione pulita tra training_params e preprocessing_params.
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
        Genera parametri finali con separazione training/preprocessing.
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
            
            # 3. Carica preprocessing params dal config
            print("📦 Step 3: Caricamento preprocessing params...")
            preprocessing_params = config.preprocessing_params.get(quality_level, {})
            print(f"✅ Preprocessing params: {preprocessing_params}")
            
            # 4. Applica quality transforms (solo training)
            print("⚙️ Step 4: Applicazione quality transforms...")
            training_params, final_preprocessing = self._apply_quality_transforms(
                config.base_params.copy(), 
                config.quality_multipliers.get(quality_level, {}),
                preprocessing_params
            )
            print(f"✅ Training params dopo quality: {training_params}")
            print(f"✅ Preprocessing params finali: {final_preprocessing}")
            
            # 5. Applica hardware scaling
            print("🔧 Step 5: Applicazione hardware scaling...")
            training_params, hw_multipliers, final_preprocessing = self._apply_hardware_scaling(
                training_params,
                config.hardware_config,
                gpu_memory_gb,
                final_preprocessing
            )
            print(f"✅ Training params dopo hardware scaling: {training_params}")
            print(f"✅ Hardware multipliers: {hw_multipliers}")
            print(f"✅ Final preprocessing params: {final_preprocessing}")
            
            # 6. Applica override manuali (solo su training params)
            print("🛠️ Step 6: Applicazione override manuali...")
            applied_overrides = {}
            if manual_overrides:
                # Filtra solo training overrides
                training_overrides = {k: v for k, v in manual_overrides.items() 
                                    if k not in {'target_width','target_height', 'target_frames'}}
                if training_overrides:
                    print(f"🛠️ Applicando training overrides: {training_overrides}")
                    training_params.update(training_overrides)
                    applied_overrides = training_overrides.copy()
                else:
                    print("🛠️ Nessun training override valido")
            else:
                print("🛠️ Nessun override manuale")
            print(f"✅ Training params dopo overrides: {training_params}")
            
            # 7. Calcola parametri derivati (se presenti)
            print("📊 Step 7: Calcolo parametri derivati...")
            if config.post_calculation:
                print("📊 Post calculation presente, eseguendo...")
                calculated = self._calculate_derived_params(training_params, config.post_calculation)
                training_params.update(calculated)
                print(f"✅ Parametri calcolati: {calculated}")
            else:
                print("📊 Nessun post calculation configurato")
            
            # 8. Valida parametri finali
            print("✅ Step 8: Validazione parametri...")
            self._validate_params(training_params, config.validation_rules)
            print("✅ Validazione completata")
            
            # 9. Pulisci parametri (arrotonda interi, etc.)
            print("🧹 Step 9: Pulizia parametri...")
            training_params = self._clean_params(training_params)
            print(f"✅ Training params finali puliti: {training_params}")
            
            # 10. Crea response con separazione
            print("📦 Step 10: Creazione response...")
            result = GeneratedParams(
                algorithm_name=config.algorithm_name,
                quality_level=quality_level,
                gpu_memory_gb=gpu_memory_gb,
                final_params=training_params,  # 🎯 SOLO parametri di training
                preprocessing_params=final_preprocessing,  # 🆕 Parametri preprocessing
                applied_multipliers=hw_multipliers,
                applied_overrides=applied_overrides,
                estimated_training_time_minutes=self._estimate_training_time(training_params),
                estimated_vram_usage_gb=self._estimate_vram_usage(training_params, gpu_memory_gb)
            )
            
            print("🎉 === FINE generate_params - SUCCESSO ===")
            print(f"🎉 Training params: {result.final_params}")
            print(f"🎉 Preprocessing params: {result.preprocessing_params}")
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
        preprocessing_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Applica moltiplicatori training e carica preprocessing params
        
        Returns:
            Tuple[training_params, preprocessing_params]
        """
        
        result = base_params.copy()
        
        print(f"🔍 DEBUG _apply_quality_transforms:")
        print(f"  base_params: {base_params}")
        print(f"  quality_mults: {quality_mults}")
        print(f"  preprocessing_params: {preprocessing_params}")

        # Applica moltiplicatori solo su training params
        for param_name, multiplier in quality_mults.items():
            if param_name in result and isinstance(result[param_name], (int, float)):
                result[param_name] = result[param_name] * multiplier
                print(f"  ⚙️ Training multiplier: {param_name} *= {multiplier}")
        
        print(f"  result finale (training): {result}")
        print(f"  preprocessing_params finale: {preprocessing_params}")

        return result, preprocessing_params
    
    def _apply_hardware_scaling(
        self, 
        params: Dict[str, Any], 
        hw_config: Any,
        gpu_memory_gb: float,
        preprocessing_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
        """
        Applica hardware scaling e gestisce target width e height
        
        Returns:
            Tuple[training_params, applied_multipliers, final_preprocessing_params]
        """

        result = params.copy()
        applied_multipliers = {}
        final_preprocessing = preprocessing_params.copy()

        print(f"🔍 DEBUG _apply_hardware_scaling INIZIATO:")
        print(f"  GPU Memory: {gpu_memory_gb} GB")
        print(f"  Baseline VRAM: {hw_config.baseline_vram_gb} GB")
        print(f"  Input training params: {params}")
        print(f"  Input preprocessing_params: {preprocessing_params}")

        # Calcola vram_factor
        vram_factor = gpu_memory_gb / hw_config.baseline_vram_gb
        print(f"  🧮 VRAM Factor = {gpu_memory_gb} / {hw_config.baseline_vram_gb} = {vram_factor:.4f}")

        # DEBUG: Log formule disponibili
        print(f"  📝 Formule hardware disponibili:")
        for param_name, formula_config in hw_config.scaling_formulas.items():
            print(f"    - {param_name}: {formula_config.formula} (min={formula_config.min}, max={formula_config.max})")

        # Applica formule scaling SOLO ai training parameters
        for param_name, formula_config in hw_config.scaling_formulas.items():
            if param_name in result:
                print(f"  🔧 Processando parametro training: {param_name}")
                
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
                logger.warning(f"    ❌ Parametro {param_name} NON TROVATO nei training params")

        # Gestisci target resolution con logica personalizzata
        if "target_width" in final_preprocessing and "target_height" in final_preprocessing:
            print(f"  🖼️ Gestendo target resolution...")
            
            # Resolution indicata dal preprocessing params (quality mode)
            quality_target_width = final_preprocessing["target_width"]
            quality_target_height = final_preprocessing["target_height"]
            print(f"    Quality target resolution: {quality_target_width}x{quality_target_height}")
            
            # Resolution massima permessa dalla VRAM
            hw_max_width = self._determine_auto_resolution_width(hw_config.resolution_thresholds, gpu_memory_gb)
            print(f"    HW max width calcolata: {hw_max_width}")
            
            # Prendi il valore più conservativo (width più bassa tra quality e hardware)
            final_width = min(quality_target_width, hw_max_width)
            
            # Calcola l'height mantenendo l'aspect ratio del quality mode
            aspect_ratio = quality_target_height / quality_target_width
            final_height = int(final_width * aspect_ratio)
            
            final_preprocessing["target_width"] = final_width
            final_preprocessing["target_height"] = final_height
            
            print(f"    ✅ Target resolution finale: {final_width}x{final_height}")

            print(f"  📋 RISULTATO FINALE:")
            print(f"    Training params: {result}")
            print(f"    Preprocessing params: {final_preprocessing}")
            print(f"    Moltiplicatori applicati: {applied_multipliers}")
            print(f"🏁 DEBUG _apply_hardware_scaling COMPLETATO")

            return result, applied_multipliers, final_preprocessing

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
    
    def _determine_auto_resolution_width(self, thresholds: list, gpu_memory_gb: float) -> int:
        """Determina resolution scale factor automatico basato su VRAM"""
        sorted_thresholds = sorted(thresholds, key=lambda x: x.vram_threshold, reverse=True)
        
        for threshold in sorted_thresholds:
            if round(gpu_memory_gb) >= threshold.vram_threshold:
                return threshold.target_width
        
        return 1280  # Fallback conservativo (risoluzione molto bassa)
    
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
        """Stima uso VRAM basato  su preprocessing_params"""
        base_usage = gpu_memory_gb * 0.8
        # Nota: resolution_scale_factor ora è in preprocessing_params, non in training params
        # Questa funzione può essere semplificata o ricevere preprocessing_params come parametro
        return base_usage * 0.8  # Stima conservativa
 