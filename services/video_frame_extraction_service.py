"""
Video Frame Extraction Service
Servizio atomico per l'estrazione e il preprocessing dei frame video
"""
import os
import cv2
import subprocess
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FrameExtractionParams:
    """Parametri per l'estrazione frame"""
    target_width: int
    target_height: int
    target_frame_count: int = 200
    selection_method: str = "best-n"
    min_buffer: str = "1"


@dataclass
class FrameExtractionResult:
    """Risultato dell'operazione di estrazione frame"""
    success: bool
    extracted_frame_count: int
    frame_files: list
    extraction_params: Dict[str, Any]
    error_message: Optional[str] = None
    processing_stats: Optional[Dict[str, Any]] = None


class VideoFrameExtractionService:
    """
    Servizio per l'estrazione atomica dei frame video con preprocessing intelligente.
    Coordina tutte le operazioni necessarie per preparare i frame per COLMAP.
    """
    
    def __init__(self):
        pass
    
    def extract_frames(
        self, 
        video_path: str, 
        output_directory: str, 
        extraction_params: FrameExtractionParams
    ) -> FrameExtractionResult:
        """
        Operazione atomica per l'estrazione completa dei frame video.
        
        Args:
            video_path: Percorso del video sorgente
            output_directory: Directory di output per i frame estratti
            extraction_params: Parametri di estrazione configurabili
            
        Returns:
            FrameExtractionResult con esito dell'operazione e statistiche
        """
        try:
            print(f"🎬 Starting atomic frame extraction for: {video_path}")
            
            # Step 1: Validazione input
            if not self._validate_inputs(video_path, output_directory):
                return FrameExtractionResult(
                    success=False,
                    extracted_frame_count=0,
                    frame_files=[],
                    extraction_params={},
                    error_message="Input validation failed"
                )
            
            # Step 2: Analisi video e calcolo parametri ottimali
            video_analysis = self._analyze_video(video_path)
            print(f"📊 Video analysis: {video_analysis}")
            
            # Step 3: Calcolo parametri di estrazione adattivi
            optimized_params = self._calculate_extraction_parameters(
                video_analysis, extraction_params
            )
            print(f"🎯 Optimized extraction parameters: {optimized_params}")
            
            # Step 4: Preparazione directory output
            os.makedirs(output_directory, exist_ok=True)
            
            # Step 5: Esecuzione estrazione con Sharp-Frames
            extraction_result = self._execute_sharp_frames_extraction(
                video_path, output_directory, optimized_params
            )
            
            if not extraction_result['success']:
                return FrameExtractionResult(
                    success=False,
                    extracted_frame_count=0,
                    frame_files=[],
                    extraction_params=optimized_params,
                    error_message=extraction_result['error_message']
                )
            
            # Step 6: Validazione output e conteggio frame
            frame_files = self._validate_and_count_frames(output_directory)
            
            # Step 7: Generazione statistiche finali
            processing_stats = {
                'video_analysis': video_analysis,
                'optimized_params': optimized_params,
                'sharp_frames_output': extraction_result.get('stdout', ''),
                'sharp_frames_warnings': extraction_result.get('stderr', '')
            }
            
            print(f"✅ Frame extraction completed successfully: {len(frame_files)} frames")
            
            return FrameExtractionResult(
                success=True,
                extracted_frame_count=len(frame_files),
                frame_files=frame_files,
                extraction_params=optimized_params,
                processing_stats=processing_stats
            )
            
        except Exception as e:
            print(f"❌ Atomic frame extraction failed: {str(e)}")
            return FrameExtractionResult(
                success=False,
                extracted_frame_count=0,
                frame_files=[],
                extraction_params={},
                error_message=f"Extraction failed: {str(e)}"
            )
    
    def _validate_inputs(self, video_path: str, output_directory: str) -> bool:
        """Validazione degli input prima dell'elaborazione"""
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False
        
        if not os.access(os.path.dirname(output_directory), os.W_OK):
            print(f"❌ Output directory not writable: {output_directory}")
            return False
            
        # Verifica che il file sia un video valido
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video file: {video_path}")
            cap.release()
            return False
        cap.release()
        
        return True
    
    def _analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analisi completa delle caratteristiche video per ottimizzazione parametri.
        Combina le logiche di calculate_target_width e calculate_extraction_fps.
        """
        cap = cv2.VideoCapture(video_path)
        
        # Estrazione metadati video
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        cap.release()
        
        # Determinazione orientamento
        is_portrait = original_height > original_width
        orientation = 'Portrait' if is_portrait else 'Landscape'
        
        analysis = {
            'original_width': original_width,
            'original_height': original_height,
            'video_fps': video_fps,
            'total_frames': total_frames,
            'duration_seconds': duration,
            'is_portrait': is_portrait,
            'orientation': orientation,
            'aspect_ratio': original_width / original_height if original_height > 0 else 1.0
        }
        
        print(f"📺 Original resolution: {original_width}x{original_height}")
        print(f"📐 Orientation: {orientation}")
        print(f"🎞️ Duration: {duration:.2f}s | 🎥 Original FPS: {video_fps:.2f}")
        
        return analysis
    
    def _calculate_extraction_parameters(
        self, 
        video_analysis: Dict[str, Any], 
        extraction_params: FrameExtractionParams
    ) -> Dict[str, Any]:
        """
        Calcola parametri ottimizzati per l'estrazione basati sull'analisi video.
        Integra la logica di calculate_target_width e calculate_extraction_fps.
        """
        
        # ===== CALCOLO WIDTH ADATTIVA (da calculate_target_width) =====
        target_width = extraction_params.target_width
        target_height = extraction_params.target_height
        is_portrait = video_analysis['is_portrait']
        
        # Per portrait, target_height diventa la width effettiva
        final_width = target_height if is_portrait else target_width
        
        print(f"🎯 Target resolution: {target_width}x{target_height}")
        print(f"📏 Final width (adapted): {final_width}")
        
        # ===== CALCOLO FPS ESTRAZIONE (da calculate_extraction_fps) =====
        duration = video_analysis['duration_seconds']
        video_fps = video_analysis['video_fps']
        target_frame_count = extraction_params.target_frame_count
        
        print(f"🎯 Target frame count: {target_frame_count}")
        
        if duration > 0:
            extraction_fps = target_frame_count / duration
            extraction_fps = min(extraction_fps, video_fps)  # Non può superare FPS originale
            extraction_fps = max(0.5, extraction_fps)  # Soglia minima anti-sottocampionamento
        else:
            extraction_fps = 1.0  # Fallback sicuro
            
        extraction_fps_rounded = round(extraction_fps)
        
        print(f"⏱️ Extraction FPS: {extraction_fps:.2f} (rounded: {extraction_fps_rounded})")
        
        return {
            'final_width': final_width,
            'extraction_fps': extraction_fps_rounded,
            'selection_method': extraction_params.selection_method,
            'min_buffer': extraction_params.min_buffer,
            'target_frame_count': target_frame_count,
            # Parametri derivati per debug/stats
            'calculated_extraction_fps_precise': extraction_fps,
            'width_adaptation_applied': is_portrait
        }
    
    def _execute_sharp_frames_extraction(
        self, 
        video_path: str, 
        output_directory: str, 
        optimized_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Esecuzione atomica del comando Sharp-Frames con parametri ottimizzati.
        """
        
        # Costruzione comando Sharp-Frames
        cmd = [
            "sharp-frames",
            video_path,
            output_directory,
            "--selection-method", optimized_params['selection_method'],
            "--min-buffer", optimized_params['min_buffer'],
            "--fps", str(optimized_params['extraction_fps'])
        ]
        
        # Aggiunta parametro width se necessario
        final_width = optimized_params.get('final_width')
        if final_width is not None:
            cmd.extend(["--width", str(final_width)])
        
        print(f"🔧 Executing Sharp-Frames: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ Sharp-frames completed successfully")
            print(f"📝 stdout: {result.stdout}")
            
            if result.stderr:
                print(f"⚠️ Sharp-frames warnings: {result.stderr}")
            
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Sharp-frames failed with exit code {e.returncode}")
            print(f"📝 stdout: {e.stdout}")
            print(f"🔴 stderr: {e.stderr}")
            
            return {
                'success': False,
                'stdout': e.stdout,
                'stderr': e.stderr,
                'return_code': e.returncode,
                'error_message': f"Sharp-frames error: {e.stderr}"
            }
    
    def _validate_and_count_frames(self, output_directory: str) -> list:
        """
        Validazione dell'output e conteggio dei frame estratti.
        """
        frame_files = []
        
        if os.path.exists(output_directory):
            for filename in sorted(os.listdir(output_directory)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(output_directory, filename)
                    frame_files.append(full_path)
        
        print(f"📊 Extracted frames validated: {len(frame_files)}")
        
        if len(frame_files) == 0:
            raise Exception("No frames were extracted - check Sharp-Frames configuration")
        
        return frame_files


# ===== FACTORY FUNCTION PER INTEGRAZIONE FACILE =====
def create_frame_extraction_service() -> VideoFrameExtractionService:
    """Factory function per creare il service"""
    return VideoFrameExtractionService()