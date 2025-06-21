#!/usr/bin/env python3
"""
Safe preprocessor che fallback su copia semplice se SAM2 non disponibile
"""

import os
import sys
import subprocess
from pathlib import Path

def ensure_sam2_installed():
    """Installa SAM2 a runtime se non disponibile."""
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2 già disponibile")
        return True
    except ImportError:
        print("📦 SAM2 non trovato, installazione a runtime...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/facebookresearch/segment-anything-2.git"
            ])
            
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print("✅ SAM2 installato con successo!")
            return True
            
        except Exception as e:
            print(f"❌ Fallita installazione SAM2: {e}")
            return False

def get_preprocessor_class():
    """Ottiene la classe preprocessor, installando SAM2 se necessario."""
    
    if not ensure_sam2_installed():
        print("⚠️  SAM2 non disponibile, usando preprocessor dummy")
        return DummyPreprocessor
    
    try:
        from utils.gaussian_frame_segmentator import GaussianSplattingSegmentator
        return GaussianSplattingSegmentator
    except ImportError:
        print("⚠️  gaussian_frame_preprocessor non trovato, usando dummy")
        return DummyPreprocessor

class DummyPreprocessor:
    """Preprocessor dummy se SAM2 non è disponibile."""
    
    def __init__(self, *args, **kwargs):
        print("⚠️  Usando preprocessor dummy (copia semplice)")
    
    def process_dataset(self, input_path, output_path, **kwargs):
        """Copia semplice delle immagini senza preprocessing."""
        import shutil
        from pathlib import Path
        from tqdm import tqdm
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Trova immagini
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        image_files = []
        for ext in valid_extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        image_files = list(set(image_files))
        
        print(f"📋 Copia semplice di {len(image_files)} immagini...")
        
        # Copia file
        copied = 0
        for img_file in tqdm(image_files, desc="Copying images"):
            try:
                output_file = output_path / img_file.name
                shutil.copy2(img_file, output_file)
                copied += 1
            except Exception as e:
                print(f"❌ Errore copia {img_file}: {e}")
        
        return {
            "total_images": len(image_files),
            "processed_images": copied,
            "failed_images": len(image_files) - copied,
            "average_coverage": 1.0,
            "method": "dummy_copy",
            "preprocessing_used": False
        }

def preprocess_dataset_for_training(
    input_path: str,
    output_path: str,
    method: str = "center_point",
    removal_level: str = "moderate",
    use_transparent_bg: bool = True,
    model_size: str = "base_plus"
) -> dict:
    """
    Preprocessing sicuro con fallback automatico.
    """
    
    print(f"🚀 Avvio preprocessing dataset...")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    try:
        # Verifica che ci siano immagini da processare
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Directory input non trovata: {input_path}")
        
        # Conta immagini disponibili
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        image_files = []
        for ext in valid_extensions:
            image_files.extend(input_path_obj.rglob(f"*{ext}"))
            image_files.extend(input_path_obj.rglob(f"*{ext.upper()}"))
        
        image_files = list(set(image_files))
        
        if len(image_files) == 0:
            print(f"⚠️  Nessuna immagine trovata in {input_path}")
            return {
                "total_images": 0,
                "processed_images": 0,
                "failed_images": 0,
                "average_coverage": 0.0,
                "method": "no_images",
                "preprocessing_used": False
            }
        
        print(f"📊 Trovate {len(image_files)} immagini da processare")
        
        # Ottieni preprocessor (vero o dummy)
        PreprocessorClass = get_preprocessor_class()
        
        if PreprocessorClass.__name__ == "DummyPreprocessor":
            # Usa dummy preprocessor (copia semplice)
            preprocessor = PreprocessorClass()
            stats = preprocessor.process_dataset(
                input_path=input_path,
                output_path=output_path
            )
        else:
            # Usa vero preprocessor con SAM2
            print(f"🤖 Utilizzo preprocessing intelligente con SAM2")
            preprocessor = PreprocessorClass(
                model_size=model_size,
                cache_dir="/code/.cache/sam2_models"
            )
            
            background_color = "transparent" if use_transparent_bg else None
            
            stats = preprocessor.process_dataset(
                input_path=input_path,
                output_path=output_path,
                method=method,
                removal_level=removal_level,
                background_color=background_color,
                preserve_structure=True,
                save_masks=False
            )
            stats["preprocessing_used"] = True
        
        print(f"\n✅ Preprocessing completato!")
        print(f"📊 Successo: {stats['processed_images']}/{stats['total_images']} immagini")
        print(f"🎯 Metodo: {stats.get('method', 'unknown')}")
        
        return stats
        
    except Exception as e:
        print(f"❌ Errore durante preprocessing: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback finale: copia semplice
        print("🔄 Fallback su copia semplice...")
        try:
            dummy = DummyPreprocessor()
            return dummy.process_dataset(input_path, output_path)
        except Exception as e2:
            print(f"❌ Anche il fallback è fallito: {e2}")
            return {
                "total_images": 0,
                "processed_images": 0,
                "failed_images": 0,
                "average_coverage": 0.0,
                "method": "failed",
                "preprocessing_used": False,
                "error": str(e)
            }#!/usr/bin/env python3
"""
Versione sicura del preprocessor che installa SAM2 a runtime se necessario
"""

import os
import sys
import subprocess
from pathlib import Path

def ensure_sam2_installed():
    """Installa SAM2 a runtime se non disponibile."""
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2 già disponibile")
        return True
    except ImportError:
        print("📦 SAM2 non trovato, installazione a runtime...")
        
        try:
            # Installa SAM2
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/facebookresearch/segment-anything-2.git"
            ])
            
            # Verifica installazione
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print("✅ SAM2 installato con successo!")
            return True
            
        except Exception as e:
            print(f"❌ Fallita installazione SAM2: {e}")
            return False

def get_preprocessor_class():
    """Ottiene la classe preprocessor, installand SAM2 se necessario."""
    
    if not ensure_sam2_installed():
        print("⚠️  SAM2 non disponibile, usando preprocessor dummy")
        return DummyPreprocessor
    
    # Importa la vera classe
    from utils.gaussian_frame_segmentator import GaussianSplattingSegmentator
    return GaussianSplattingSegmentator

class DummyPreprocessor:
    """Preprocessor dummy se SAM2 non è disponibile."""
    
    def __init__(self, *args, **kwargs):
        print("⚠️  Usando preprocessor dummy (copia semplice)")
    
    def process_dataset(self, input_path, output_path, **kwargs):
        """Copia semplice delle immagini senza preprocessing."""
        import shutil
        from pathlib import Path
        from tqdm import tqdm
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Trova immagini
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        image_files = []
        for ext in valid_extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        image_files = list(set(image_files))
        
        # Copia file
        for img_file in tqdm(image_files, desc="Copying images"):
            output_file = output_path / img_file.name
            shutil.copy2(img_file, output_file)
        
        return {
            "total_images": len(image_files),
            "processed_images": len(image_files),
            "failed_images": 0,
            "average_coverage": 1.0,
            "method": "dummy_copy"
        }

def preprocess_dataset_for_training(
    input_path: str,
    output_path: str,
    method: str = "center_point",
    removal_level: str = "moderate",
    use_transparent_bg: bool = True,
    model_size: str = "base_plus"
) -> dict:
    """
    Versione sicura del preprocessing che fallback su copia semplice.
    """
    
    print(f"🚀 Avvio preprocessing dataset...")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    try:
        # Ottieni preprocessor (vero o dummy)
        PreprocessorClass = get_preprocessor_class()
        
        if PreprocessorClass.__name__ == "DummyPreprocessor":
            # Usa dummy preprocessor
            preprocessor = PreprocessorClass()
            stats = preprocessor.process_dataset(
                input_path=input_path,
                output_path=output_path
            )
        else:
            # Usa vero preprocessor
            preprocessor = PreprocessorClass(
                model_size=model_size,
                cache_dir="/code/.cache/sam2_models"
            )
            
            background_color = "transparent" if use_transparent_bg else None
            
            stats = preprocessor.process_dataset(
                input_path=input_path,
                output_path=output_path,
                method=method,
                removal_level=removal_level,
                background_color=background_color,
                preserve_structure=True,
                save_masks=False
            )
        
        print(f"\n✅ Preprocessing completato!")
        print(f"📊 Successo: {stats['processed_images']}/{stats['total_images']} immagini")
        
        return stats
        
    except Exception as e:
        print(f"❌ Errore durante preprocessing: {e}")
        
        # Fallback: copia semplice
        print("🔄 Fallback su copia semplice...")
        dummy = DummyPreprocessor()
        return dummy.process_dataset(input_path, output_path)

def main():
    """Entry point standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe Preprocessing per Gaussian Splatting")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="center_point")
    parser.add_argument("--removal", default="moderate")
    parser.add_argument("--model", default="base_plus")
    parser.add_argument("--no-transparent", action="store_true")
    
    args = parser.parse_args()
    
    stats = preprocess_dataset_for_training(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        removal_level=args.removal,
        use_transparent_bg=not args.no_transparent,
        model_size=args.model
    )
    
    print(f"🎉 Completed with method: {stats.get('method', 'unknown')}")

if __name__ == "__main__":
    main()