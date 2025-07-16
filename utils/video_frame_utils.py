import cv2
import numpy as np
import os
from collections import deque
import time
from pymediainfo import MediaInfo
import hashlib
from glob import glob
from skimage.metrics import structural_similarity as compare_ssim
import re
from tqdm import tqdm

class Image:
    def __init__(self, id, qvec, tvec, camera_id, name, xys=None, point3D_ids=None):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys if xys is not None else np.zeros((0, 2))
        self.point3D_ids = point3D_ids if point3D_ids is not None else np.zeros(0, dtype=np.int64)

class FrameExtractor:

    def calculate_target_width(self,video_path, scale_factor=1, min_width=1280, min_portrait_width=720):
        """
        Calcola la width finale dopo downscaling, rispettando minimi in base all'orientamento.
        - Landscape: min_width
        - Portrait: min_portrait_width
        L'aspect ratio verrà mantenuto esternamente.
        
        :param video_path: Percorso del video
        :param scale_factor: Fattore di downscaling (1, 2, 4, 8, ...)
        :param min_width: Width minima per video orizzontali
        :param min_portrait_width: Width minima per video verticali
        :return: Width finale (int)
        """
        cap = cv2.VideoCapture(video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        is_portrait = original_height > original_width
        min_required_width = min_portrait_width if is_portrait else min_width

        print(f"📺 Original resolution: {original_width}x{original_height}")
        print(f"📐 Orientation: {'Portrait' if is_portrait else 'Landscape'}")
        print(f"🔢 Requested scale factor: {scale_factor}")

        scaled_width = original_width // scale_factor
        print(f"📉 Scaled width: {scaled_width} (min allowed: {min_required_width})")

        if scaled_width < min_required_width:
            print(f"⚠️ Scaled width too small. Adjusting...")

            max_safe_scale = original_width // min_required_width

            # Potenza di 2 più vicina verso il basso
            def nearest_lower_power_of_two(n):
                power = 1
                while power * 2 <= n:
                    power *= 2
                return power

            adjusted_scale_factor = nearest_lower_power_of_two(max_safe_scale)
            print(f"🔁 Adjusted scale factor: {adjusted_scale_factor}")

            scaled_width = original_width // adjusted_scale_factor
            print(f"📏 Final adjusted width: {scaled_width}")

        return scaled_width
    
    def calculate_extraction_fps(self,video_path, target_frame_count=180):
        """
        Calcola l'FPS di estrazione per generare ~150-200 frame da dare a COLMAP.
        Utile per Gaussian Splatting o MVS, con copertura temporale regolare.
        
        :param video_path: Percorso del video
        :param target_frame_count: Numero target di frame (default 180 per ~1 min)
        :return: FPS di estrazione (float)
        """
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        cap.release()

        print(f"🎞️ Duration: {duration:.2f}s | 🎯 Target frames: {target_frame_count} | 🎥 Original FPS: {video_fps:.2f}")

        extraction_fps = target_frame_count / duration
        extraction_fps = min(extraction_fps, video_fps)
        extraction_fps = max(0.5, extraction_fps)  # minimo tecnico per non sotto-campionare troppo

        print(f"⏱️ Extraction FPS: {extraction_fps:.2f}")

        return round(extraction_fps)
    
   
    
    def filter_similar_frames(self,input_dir, output_dir, similarity_threshold=0.80):
        """
        Filtra i frame ridondanti da una sequenza ordinata di immagini, conservando solo quelli
        con copertura visiva inferiore alla soglia di similarità (SSIM).

        Args:
            image_paths (List[str]): Lista di path alle immagini ordinate temporalmente.
            output_dir (str): Cartella dove salvare le immagini non ridondanti.
            similarity_threshold (float): Soglia SSIM oltre la quale un frame è considerato troppo simile (default: 0.80).

        Returns:
            List[str]: Lista dei path delle immagini salvate.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Ordina i frame per nome numerico
        image_paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"🔎 Frame trovati in input_dir: {len(image_paths)}")
        print(f"📂 Frame ordinati per indice: {len(image_paths)} frame totali")
        
        kept_frames = []
        prev_gray = None

        for path in tqdm(image_paths, desc="Comparing frames"):
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: unable to read image at {path}, skipping.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                # Salva sempre il primo frame
                out_path = os.path.join(output_dir, f"frame_{len(kept_frames):04d}.jpg")
                cv2.imwrite(out_path, img)
                kept_frames.append(out_path)
                prev_gray = gray
                continue

            # Calcola SSIM tra il frame corrente e quello precedente tenuto
            similarity, _ = compare_ssim(prev_gray, gray, full=True)

            if similarity < similarity_threshold:
                out_path = os.path.join(output_dir, f"frame_{len(kept_frames):04d}.jpg")
                cv2.imwrite(out_path, img)
                kept_frames.append(out_path)
                prev_gray = gray
            else:
                print(f"Frame {path} skipped (SSIM = {similarity:.3f})")

        return kept_frames

    
    

   
    
    def generate_video_hash(self, file_path):
        """
        Generate a hash of the video file.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            SHA256 hash of the video file
        """
        # Create a SHA256 hash object
        sha256_hash = hashlib.sha256()
        
        # Open the video file in binary mode
        with open(file_path, "rb") as f:
            # Read the file in 4kB blocks
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        # Return the hash in hexadecimal format
        return sha256_hash.hexdigest()
    
    # Helper methods
    
    
    
    
    

