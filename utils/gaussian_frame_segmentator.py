import sys
import os
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# IMPORTA PyTorch PRIMA delle extension C++ (importante per evitare _C errors)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Per GroundingDINO
try:
    import groundingdino
    from groundingdino.models import build_model
    from groundingdino.util import box_ops
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util.inference import annotate, load_image, predict
    print("✅ GroundingDINO imported successfully")
except ImportError as e:
    print(f"❌ GroundingDINO import error: {e}")
    raise

# Per SAM2 - IMPORTA CON GESTIONE ERRORI
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("✅ SAM2 imported successfully")
except ImportError as e:
    print(f"❌ SAM2 import error: {e}")
    print("Trying alternative import path...")
    try:
        sys.path.append('/tmp/sam2')
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2 imported successfully (alternative path)")
    except ImportError as e2:
        print(f"❌ SAM2 alternative import also failed: {e2}")
        raise

class GroundingDINOSAMSegmentator:
    def __init__(self, 
                 grounding_dino_config_path=None,
                 grounding_dino_checkpoint_path=None,
                 sam2_checkpoint_path=None,
                 sam2_model_id="sam2_hiera_b+"):  # Usa model_id invece di config path
        
        # Path default per Docker/locale
        self.base_models_path = "/code/models" if os.path.exists("/code/models") else "models"
        
        # GroundingDINO paths
        if grounding_dino_config_path is None:
            grounding_dino_config_path = os.path.join(self.base_models_path, "GroundingDINO_SwinT_OGC.py")
        if grounding_dino_checkpoint_path is None:
            grounding_dino_checkpoint_path = os.path.join(self.base_models_path, "groundingdino_swint_ogc.pth")
            
        # SAM2 checkpoint - SENZA config file
        if sam2_checkpoint_path is None:
            sam2_checkpoint_path = os.path.join(self.base_models_path, "sam2_hiera_base_plus.pt")
        
        # Verifica esistenza modelli
        self._verify_models_exist(grounding_dino_config_path, grounding_dino_checkpoint_path, sam2_checkpoint_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Inizializza GroundingDINO
        print("Loading GroundingDINO...")
        self.grounding_model = self._load_grounding_dino(grounding_dino_config_path, grounding_dino_checkpoint_path)
        
        # Inizializza SAM2 - USA SOLO IL MODEL_ID, SENZA CONFIG FILE
        print(f"Loading SAM2 ({sam2_model_id})...")
        self.sam2_model = build_sam2(sam2_model_id, sam2_checkpoint_path, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        print("✅ Models loaded successfully!")
    
    def _verify_models_exist(self, grounding_config, grounding_checkpoint, sam2_checkpoint):
        """Verifica che i modelli essenziali esistano"""
        models_to_check = {
            "GroundingDINO Config": grounding_config,
            "GroundingDINO Checkpoint": grounding_checkpoint,
            "SAM2 Checkpoint": sam2_checkpoint
        }
        
        missing_models = []
        for name, path in models_to_check.items():
            if not os.path.exists(path):
                missing_models.append(f"{name}: {path}")
        
        if missing_models:
            print(f"❌ Modelli mancanti:")
            for missing in missing_models:
                print(f"  - {missing}")
            
            # Mostra modelli disponibili
            if os.path.exists(self.base_models_path):
                print(f"\n📁 File disponibili in {self.base_models_path}:")
                for file in os.listdir(self.base_models_path):
                    full_path = os.path.join(self.base_models_path, file)
                    if os.path.isfile(full_path):
                        size = os.path.getsize(full_path) / (1024*1024)  # MB
                        print(f"  - {file} ({size:.1f} MB)")
            
            raise FileNotFoundError(f"Modelli mancanti: {', '.join(missing_models)}")
        
        print("✅ Tutti i modelli sono presenti")
    
    def _load_grounding_dino(self, config_path, checkpoint_path):
        """Carica GroundingDINO"""
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(f"GroundingDINO load result: {load_res}")
        model.eval()
        return model
    
    def segment_with_text_prompt(self, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Segmenta oggetto usando prompt testuale
        
        Args:
            image_path: Path dell'immagine
            text_prompt: Descrizione dell'oggetto (es. "blue playground toy", "potted plant", "R2D2 robot")
            box_threshold: Soglia confidence per detection boxes
            text_threshold: Soglia confidence per text matching
        """
        
        try:
            print(f"Processing: {os.path.basename(image_path)}")
            print(f"Prompt: '{text_prompt}'")
            
            # 1. CARICA IMMAGINE
            image_source, image = load_image(image_path)
            
            # 2. GROUNDING DINO - Trova oggetto con descrizione testuale
            boxes, logits, phrases = predict(
                model=self.grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            if len(boxes) == 0:
                print(f"⚠️ Nessun oggetto trovato per prompt: '{text_prompt}'")
                return {
                    'success': False,
                    'error': f'No objects detected for prompt: {text_prompt}'
                }
            
            print(f"✓ GroundingDINO trovato {len(boxes)} oggetti")
            
            # 3. SCEGLI IL BOX CON CONFIDENCE PIÙ ALTA + FILTRI DIMENSIONALI
            valid_boxes = []
            valid_logits = []
            
            H, W = image_source.shape[:2]
            
            for i, (box, logit) in enumerate(zip(boxes, logits)):
                # Converti box per analisi dimensionale
                box_xyxy = box_ops.box_cxcywh_to_xyxy(box) * torch.Tensor([W, H, W, H])
                x1, y1, x2, y2 = box_xyxy.int().tolist()
                
                # Calcola metriche
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                area_ratio = box_area / (W * H)
                aspect_ratio = box_width / box_height if box_height > 0 else 0
                
                # Applica filtri se definiti
                area_ok = True
                aspect_ok = True
                
                if hasattr(self, 'min_area_ratio') and hasattr(self, 'max_area_ratio'):
                    area_ok = self.min_area_ratio <= area_ratio <= self.max_area_ratio
                    
                if hasattr(self, 'min_aspect_ratio') and hasattr(self, 'max_aspect_ratio'):
                    aspect_ok = self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
                
                if area_ok and aspect_ok:
                    valid_boxes.append(box)
                    valid_logits.append(logit)
                    print(f"✓ Box valido: area={area_ratio:.1%}, aspect={aspect_ratio:.2f}, conf={logit:.3f}")
                else:
                    print(f"⚠️ Box scartato: area={area_ratio:.1%}, aspect={aspect_ratio:.2f} (filtri)")
            
            if len(valid_boxes) == 0:
                print(f"⚠️ Nessun oggetto valido dopo filtri dimensionali")
                return {
                    'success': False,
                    'error': 'No valid objects after dimensional filtering'
                }
            
            # Scegli il migliore tra quelli validi
            valid_logits_tensor = torch.stack(valid_logits)
            best_valid_idx = torch.argmax(valid_logits_tensor)
            best_box = valid_boxes[best_valid_idx]
            best_confidence = valid_logits_tensor[best_valid_idx].item()
            
            print(f"Best valid detection: confidence={best_confidence:.3f}")
            
            # 4. CONVERTI BOX COORDINATE per SAM2
            H, W = image_source.shape[:2]
            
            # GroundingDINO restituisce coordinate normalizzate (cx, cy, w, h)
            # Convertiamo in (x1, y1, x2, y2) assolute
            box_xyxy = box_ops.box_cxcywh_to_xyxy(best_box) * torch.Tensor([W, H, W, H])
            x1, y1, x2, y2 = box_xyxy.int().tolist()
            
            # Centro del box come prompt point per SAM2
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            prompt_point = np.array([[center_x, center_y]])
            prompt_labels = np.array([1])  # 1 = foreground
            
            print(f"SAM2 prompt point: ({center_x}, {center_y})")
            
            # 5. SAM2 - Segmentazione precisa
            self.sam2_predictor.set_image(image_source)
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=prompt_point,
                point_labels=prompt_labels,
                box=box_xyxy.numpy(),  # Usa anche il box come hint
                multimask_output=True
            )
            
            # Scegli la maschera migliore
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            mask_score = scores[best_mask_idx]
            
            print(f"✓ SAM2 segmentation score: {mask_score:.3f}")
            
            # 6. CREA OUTPUT FINALE
            result = np.zeros((H, W, 4), dtype=np.uint8)
            result[:, :, :3] = image_source  # RGB
            result[:, :, 3] = 0  # Alpha trasparente

            # CORREZIONE: Assicurati che best_mask sia boolean
            best_mask = best_mask.astype(bool)  # <-- AGGIUNGI QUESTA RIGA
            result[best_mask, 3] = 255  # Alpha opaco dove c'è l'oggetto

            # Anche il calcolo del coverage può avere problemi simili:
            coverage = np.sum(best_mask.astype(bool)) / best_mask.size  # <-- MODIFICA ANCHE QUESTA
            
            return {
                'success': True,
                'result': result,
                'mask': best_mask,
                'coverage': coverage,
                'grounding_confidence': best_confidence,
                'sam_score': mask_score,
                'detection_box': (x1, y1, x2, y2),
                'debug_data': {
                    'all_boxes': boxes,
                    'all_logits': logits,
                    'phrases': phrases,
                    'sam_masks': masks,
                    'sam_scores': scores
                }
            }
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_folder(self, input_folder, output_folder, text_prompt, 
                      box_threshold=0.3, text_threshold=0.25, save_debug=False):
        """
        Processa una cartella intera con lo stesso prompt
        
        Args:
            input_folder: Cartella con immagini input
            output_folder: Cartella output
            text_prompt: Descrizione oggetto (es. "blue playground equipment")
            box_threshold: Soglia detection
            text_threshold: Soglia text matching
            save_debug: Salva immagini debug
        """
        
        print(f"\n{'='*60}")
        print(f"GROUNDING DINO + SAM2 BATCH PROCESSING")
        print(f"Input: {input_folder}")
        print(f"Output: {output_folder}")
        print(f"Prompt: '{text_prompt}'")
        print(f"Box threshold: {box_threshold}")
        print(f"Text threshold: {text_threshold}")
        print(f"{'='*60}")
        
        # Setup cartelle
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_debug:
            debug_path = output_path / 'debug'
            debug_path.mkdir(exist_ok=True)
        
        # Trova immagini
        input_path = Path(input_folder)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
        
        image_files = sorted(image_files)
        print(f"Trovate {len(image_files)} immagini")
        
        # Statistiche
        results = {
            'total': len(image_files),
            'successful': 0,
            'failed': 0,
            'files': [],
            'failed_files': []
        }
        
        # Processa ogni immagine
        for i, image_file in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}]", end=" ")
            
            result = self.segment_with_text_prompt(
                str(image_file), 
                text_prompt, 
                box_threshold, 
                text_threshold
            )
            
            if result['success']:
                # Salva risultato
                output_filename = image_file.stem + '_segmented.png'
                output_file = output_path / output_filename
                
                result_pil = Image.fromarray(result['result'], 'RGBA')
                result_pil.save(output_file, 'PNG')
                
                results['successful'] += 1
                results['files'].append({
                    'input': str(image_file),
                    'output': str(output_file),
                    'coverage': result['coverage'],
                    'grounding_confidence': result['grounding_confidence'],
                    'sam_score': result['sam_score']
                })
                
                print(f"✅ Salvato: coverage={result['coverage']*100:.1f}%, confidence={result['grounding_confidence']:.3f}")
                
            else:
                results['failed'] += 1
                results['failed_files'].append({
                    'input': str(image_file),
                    'error': result['error']
                })
                print(f"❌ Fallito: {result['error']}")
        
        # Report finale
        print(f"\n{'='*60}")
        print(f"📊 REPORT FINALE")
        print(f"Totale: {results['total']}")
        print(f"Successi: {results['successful']} ({results['successful']/results['total']*100:.1f}%)")
        print(f"Fallimenti: {results['failed']} ({results['failed']/results['total']*100:.1f}%)")
        
        if results['failed'] > 0:
            print(f"\n❌ Immagini fallite:")
            for failed in results['failed_files']:
                print(f"  {Path(failed['input']).name}: {failed['error']}")
        
        print(f"📁 Output salvato in: {output_path}")
        print(f"{'='*60}")
        
        return results


def segment_folder(input_folder, output_folder, text_prompt, 
                  box_threshold=0.5, text_threshold=0.35, save_debug=False,
                  min_area_ratio=0.02, max_area_ratio=0.8, 
                  min_aspect_ratio=0.3, max_aspect_ratio=4.0):
    """
    Funzione principale per segmentazione automatica - Versione semplificata
    
    Args:
        input_folder (str): Cartella con immagini input
        output_folder (str): Cartella output per immagini segmentate
        text_prompt (str): Descrizione oggetto (es. "blue spring rider toy")
        box_threshold (float): Soglia confidence detection (0.1-0.9, default: 0.5)
        text_threshold (float): Soglia text matching (0.1-0.9, default: 0.35)
        save_debug (bool): Salva immagini debug (default: False)
        min_area_ratio (float): Area minima oggetto (% immagine, default: 0.02)
        max_area_ratio (float): Area massima oggetto (% immagine, default: 0.8)
        min_aspect_ratio (float): Aspect ratio minimo (default: 0.3)
        max_aspect_ratio (float): Aspect ratio massimo (default: 4.0)
    
    Returns:
        dict: Risultati processamento con statistiche
    """
    
    print(f"🚀 Segmentazione automatica avviata")
    print(f"📁 Input: {input_folder}")
    print(f"📁 Output: {output_folder}")
    print(f"🎯 Prompt: '{text_prompt}'")
    print(f"⚙️ Parametri: box_th={box_threshold}, text_th={text_threshold}")
    
    try:
        # Inizializza segmentatore - SENZA CONFIG YAML
        segmentator = GroundingDINOSAMSegmentator()
        
        # Aggiungi filtri dimensionali al segmentatore
        segmentator.min_area_ratio = min_area_ratio
        segmentator.max_area_ratio = max_area_ratio
        segmentator.min_aspect_ratio = min_aspect_ratio
        segmentator.max_aspect_ratio = max_aspect_ratio
        
        # Processa cartella
        results = segmentator.process_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            save_debug=save_debug
        )
        
        print(f"✅ Completato: {results['successful']}/{results['total']} successi")
        return results
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return {
            'success': False,
            'error': str(e),
            'total': 0,
            'successful': 0,
            'failed': 0
        }


# Test main
if __name__ == "__main__":
    # Test veloce per verificare che tutto funzioni
    try:
        segmentator = GroundingDINOSAMSegmentator()
        print("✅ Docker setup completato correttamente!")
    except Exception as e:
        print(f"❌ Errore nel setup: {e}")
        import traceback
        traceback.print_exc()