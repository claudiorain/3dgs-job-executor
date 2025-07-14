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
    
    def extract_frame_number(self,filename):
        match = re.search(r'(\d+)',filename)
        return int(match.group(1)) if match else float('inf')
    
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

    
    def get_target_width_from_video(self,video_path, target_max_size=1920):
        """
        Analizza un video e restituisce la width target per sharp-frames-python
        
        Args:
            video_path: Percorso del file video
            target_max_size: Dimensione massima target (default 1920 per Full HD)
        
        Returns:
            int: Width target per --width parameter, o None se no resize needed
        """
        try:
            # Apri il video
            cap = cv2.VideoCapture(video_path)
            
            # Ottieni dimensioni
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Chiudi il video
            cap.release()
            
            # Calcola target width
            max_dim = max(width, height)
            
            # Se già sotto target, non serve resize
            if max_dim <= target_max_size:
                return None
            
            # Calcola scale factor e target width
            scale_factor = target_max_size / max_dim
            target_width = int(width * scale_factor)
            
            return target_width
        
        except Exception as e:
            print(f"Errore nell'analisi del video {video_path}: {e}")
            return None
        
    def estimate_rotation_from_flow(self,prev_gray, curr_gray):
        """
        Stima rotazione camera da optical flow
        """
        # Rileva feature points
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, 
                                        qualityLevel=0.01, minDistance=10)
        
        if corners is None or len(corners) < 10:
            return 0.0
        
        # Calcola optical flow
        next_corners, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, corners, None)
        
        # Filtra punti validi
        good_old = corners[status == 1]
        good_new = next_corners[status == 1]
        
        if len(good_old) < 10:
            return 0.0
        
        # Calcola movimento medio orizzontale (rotation indicator)
        horizontal_movement = np.mean(good_new[:, 0] - good_old[:, 0])
        
        # Converti in gradi (euristica)
        image_width = prev_gray.shape[1]
        # Assumi FOV ~60° per smartphone
        degrees_per_pixel = 60.0 / image_width
        rotation_degrees = abs(horizontal_movement * degrees_per_pixel)
        
        return rotation_degrees
    
    def analyze_camera_rotation(self,video_path, sample_count=10):
        """
        Analisi rotazione semplificata
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = np.linspace(0, frame_count-1, sample_count, dtype=int)
        
        rotations = []
        prev_gray = None
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # ✅ Passa entrambi i frame
                rotation = self.estimate_rotation_from_flow(prev_gray, gray)
                rotations.append(rotation)
            
            prev_gray = gray
        
        cap.release()
        
        if not rotations:
            return 6.0  # Fallback
        
        # Calcola velocità rotazione
        frames_between_samples = frame_count / sample_count
        time_between_samples = frames_between_samples / fps
        avg_rotation_per_sample = np.mean(rotations)
        
        return avg_rotation_per_sample / time_between_samples

    

    def calculate_fps_for_colmap_overlap(self,video_path, target_overlap=0.7):
        """
        Calcola FPS per garantire 60-80% overlap tra frame consecutivi
        """
        # Analizza movimento camera (rotation speed)
        rotation_speed = self.analyze_camera_rotation(video_path)  # degrees/second
        
        # Per 70% overlap in scene 360°:
        # Ogni frame deve vedere ~70% della scena del precedente
        # → Movimento massimo = 30% del FOV tra frame consecutivi
        
        camera_fov = 60  # gradi (typical smartphone)
        max_rotation_per_frame = camera_fov * (1 - target_overlap)  # ~18 gradi
        
        # FPS necessario
        required_fps = rotation_speed / max_rotation_per_frame
        
        return min(required_fps, 30)  # Cap a 30fps max
    
    def calculate_gaussian_splatting_fps(self,video_analysis, quality_level='balanced'):
        """
        FPS ottimale per Gaussian Splatting considerando COLMAP overlap
        """
        
        # Base: garantire overlap COLMAP
        min_fps_for_overlap = self.calculate_fps_for_colmap_overlap(video_analysis)
        
        # Quality multipliers per coverage extra
        quality_multipliers = {
            'fast': 0.8,     # Meno denso ma ancora >60% overlap
            'balanced': 1.0, # Overlap ottimale ~70%
            'quality': 1.3,  # Overlap maggiore ~80%
        }
        
        target_fps = min_fps_for_overlap * quality_multipliers[quality_level]
    
        return round(target_fps, 1)
    
    def _calculate_downscaled_dimensions(self, original_width, original_height,
                                        downscale_factor, auto_downscale_to_fhd):
        """
        Calculate downscaled dimensions without considering rotation.
        
        Args:
            original_width: Original video width
            original_height: Original video height
            downscale_factor: Downscale factor
            auto_downscale_to_fhd: Auto downscale to Full HD
            
        Returns:
            tuple: (width, height, effective_downscale_factor)
        """
        # Apply automatic downscaling if requested
        if auto_downscale_to_fhd and (original_width > 1920 or original_height > 1080):
            width_factor = 1920 / original_width if original_width > 1920 else 1.0
            height_factor = 1080 / original_height if original_height > 1080 else 1.0
            auto_factor = min(width_factor, height_factor)
            effective_downscale_factor = downscale_factor * auto_factor
            print(f"Auto-downscaling enabled: Original resolution {original_width}x{original_height} exceeds Full HD")
            print(f"Applying downscale factor: {effective_downscale_factor:.3f}")
        else:
            effective_downscale_factor = downscale_factor
        
        # Calculate downscaled dimensions
        downscaled_width = int(original_width * effective_downscale_factor)
        downscaled_height = int(original_height * effective_downscale_factor)
        
        print(f"Downscaled dimensions: {downscaled_width}x{downscaled_height}")
        return downscaled_width, downscaled_height, effective_downscale_factor

    def extract_frames(self, video_path, output_folder,
                    threshold=0.4, min_contour_area=500, 
                    max_frames=120, min_sharpness=100,
                    enforce_temporal_distribution=True,
                    downscale_factor=1.0,
                    auto_downscale_to_fhd=True,  
                    force_rotation=None):
        """
        Extract frames from a video with intelligent selection based on movement and sharpness.
        
        Args:
            video_path: Path to the video file
            output_folder: Folder to save extracted frames
            threshold: Threshold for motion detection (0.0 to 1.0)
            min_contour_area: Minimum contour area to consider for change detection
            max_frames: Maximum number of frames to extract
            min_sharpness: Minimum sharpness value for frame selection
            enforce_temporal_distribution: Ensure frames are distributed across the video
            downscale_factor: Factor to downscale frames (1.0 = original size)
            auto_downscale_to_fhd: Automatically downscale videos larger than Full HD
            force_rotation: Force a specific rotation angle (degrees)
            
        Returns:
            Path to the first extracted frame
        """
        start_time = time.time()
        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get and normalize video orientation
        rotation_angle = self._get_normalized_rotation(video_path, force_rotation)

        # Calculate processing dimensions with rotation in mind
        downscaled_width, downscaled_height, effective_downscale_factor = self._calculate_downscaled_dimensions(
            original_width, original_height, 
            downscale_factor, auto_downscale_to_fhd
        )

        duration = total_frames / fps
        
        # Calculate minimum interval for temporal distribution
        min_frame_interval = max(1, int(total_frames / (max_frames * 2))) if enforce_temporal_distribution else 0
        print(f"Video info: {total_frames} frames, {fps} FPS, {duration:.2f}s duration")
        print(f"Original resolution: {original_width}x{original_height}, Processing resolution: {downscaled_width}x{downscaled_height}")
        print(f"Processing parameters: min interval: {min_frame_interval}")

        # Variables to keep track of processing
        extracted_count = 0
        frame_counter = 0
        last_extracted_pos = -min_frame_interval
        first_frame_path = None
        
        # Motion and viewpoint tracking
        previous_frame = None
        previous_frame_gray = None
        motion_history = deque(maxlen=10)
        
        # For exposure normalization
        exposure_stats = deque(maxlen=30)
        

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_counter += 1
            
            # Progress reporting
            if frame_counter % (total_frames // 10) == 0 or frame_counter == 1:
                print(f"Processing: {frame_counter}/{total_frames} frames ({frame_counter/total_frames*100:.1f}%)")
            
            # Convert to grayscale for feature detection and tracking
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Store exposure statistics for normalization
            exposure_stats.append(np.mean(gray))
            
            # Calculate frame sharpness
            sharpness = self.calculate_sharpness(gray)
            
            # For the first frame, initialize and always save it
            if previous_frame is None:
                previous_frame = frame.copy()
                previous_frame_gray = gray.copy()
                
                frame = cv2.resize(frame, (downscaled_width, downscaled_height)) if effective_downscale_factor != 1.0 else frame
                 # Apply rotation if needed
                if rotation_angle != 0:
                    frame = self.correct_frame_orientation(frame, rotation_angle)

                # Save the first frame
                image_filename = os.path.join(output_folder, f"frame_{extracted_count+1:04d}.jpg")
                cv2.imwrite(image_filename, frame)
                extracted_count += 1
                first_frame_path = image_filename
                last_extracted_pos = frame_counter
                
                
                continue
            
            # Check if we should process this frame (for temporal distribution)
            if enforce_temporal_distribution and (frame_counter - last_extracted_pos) < min_frame_interval:
                continue
            
            # Calculate difference from previous frame
            diff = cv2.absdiff(previous_frame_gray, gray)
            _, thresh = cv2.threshold(diff, threshold * 255, 255, cv2.THRESH_BINARY)
            
            # Find contours of changed regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
            change_percentage = sum(cv2.contourArea(c) for c in significant_contours) / (downscaled_width * downscaled_height) * 100
            
            # Add to motion history
            motion_history.append(change_percentage)
            
            # Decide if we should extract this frame
            should_extract = False
            
            # Criteria for frame extraction
            if (
                # Always require minimum sharpness
                sharpness >= min_sharpness and (
                    # Either significant motion detected
                    (change_percentage > threshold * 100) or
                    # Or moderate motion after enough time has passed
                    (change_percentage > threshold * 50 and 
                     enforce_temporal_distribution and 
                     frame_counter - last_extracted_pos >= min_frame_interval * 3)
                )
            ):
                should_extract = True
            
            # Extract the frame if conditions are met
            if should_extract and extracted_count < max_frames:

                # Update previous frame and features
                previous_frame = frame.copy()
                previous_frame_gray = gray.copy()

                image_filename = os.path.join(output_folder, f"frame_{extracted_count+1:04d}.jpg")

                frame = cv2.resize(frame, (downscaled_width, downscaled_height)) if effective_downscale_factor != 1.0 else frame
                 # Apply rotation if needed
                if rotation_angle != 0:
                    frame = self.correct_frame_orientation(frame, rotation_angle)
                

                cv2.imwrite(image_filename, frame)

                extracted_count += 1
                last_extracted_pos = frame_counter
                
                
            
            # Exit if we've extracted enough frames
            if extracted_count >= max_frames:
                print(f"Maximum number of frames ({max_frames}) extracted. Stopping.")
                break
        
        # Release resources
        cap.release()
        
        # Print extraction summary
        elapsed_time = time.time() - start_time
        print(f"Frame extraction complete. Extracted {extracted_count} frames in {elapsed_time:.2f} seconds.")
        
        return first_frame_path

    def correct_frame_orientation(self, frame, rotation_angle):
        """
        Correct the orientation of a frame based on rotation angle.
        
        Args:
            frame: The input frame
            rotation_angle: Rotation angle in degrees (0, 90, 180, 270)
            
        Returns:
            Rotated frame
        """
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # No need to apply any rotation if angle is 0
        return frame

    def calculate_sharpness(self, gray_frame):
        """
        Calculate the sharpness of an image using Laplacian variance.
        
        Args:
            gray_frame: Input grayscale frame
            
        Returns:
            Sharpness score
        """
        try:
            laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
            return laplacian.var()
        except Exception as e:
            print(f"Error calculating sharpness: {e}")
            return 0
    
    def get_video_rotation_from_metadata(self, video_path):
        """
        Retrieves the rotation angle of the video from metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Rotation angle in degrees or 0 if not found
        """
        try:
            media_info = MediaInfo.parse(video_path)
            for track in media_info.tracks:
                if track.track_type == "Video":
                    rotation = track.rotation
                    return rotation if rotation is not None else 0  # Default to 0 if no rotation info
        except Exception as e:
            print(f"Error getting video rotation: {e}")
        return 0  # Default to no rotation if error occurs
    
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
    
    def _get_normalized_rotation(self, video_path, force_rotation=None):
        """
        Get and normalize the rotation angle of the video.
        
        Args:
            video_path: Path to the video file
            force_rotation: Force a specific rotation angle
            
        Returns:
            Normalized rotation angle (0, 90, 180, 270)
        """
        if force_rotation is not None:
            rotation_angle = force_rotation
            print(f"Using forced rotation: {rotation_angle} degrees")
        else:
            rotation_angle = self.get_video_rotation_from_metadata(video_path)
            rotation_angle = float(rotation_angle)  # Ensure it's a float
            print(f"Video rotation detected from metadata: {rotation_angle} degrees")

        # Normalize the rotation angle to 0, 90, 180, or 270 degrees
        rotation_angle = rotation_angle % 360
        if rotation_angle not in [0, 90, 180, 270]:
            rotation_angle = round(rotation_angle / 90) * 90
            rotation_angle = rotation_angle % 360
            print(f"Normalized rotation angle to: {rotation_angle} degrees")
            
        return rotation_angle
    
    def _calculate_processing_dimensions(self, original_width, original_height, 
                                        rotation_angle, downscale_factor,
                                        auto_downscale_to_fhd):
        """
        Calculate processing dimensions with rotation and downscaling in mind.
        """
        # Consider rotation for dimensions
        width, height = original_width, original_height
        if rotation_angle in [90, 270]:
            # Swap width and height for calculations if rotated 90 or 270 degrees
            width, height = height, width
            print(f"Adjusted dimensions for rotation: {width}x{height}")
        
        # Apply automatic downscaling with the correct orientation in mind
        if auto_downscale_to_fhd and (width > 1920 or height > 1080):
            width_factor = 1920 / width if width > 1920 else 1.0
            height_factor = 1080 / height if height > 1080 else 1.0
            auto_factor = min(width_factor, height_factor)
            downscale_factor = downscale_factor * auto_factor
            print(f"Auto-downscaling enabled: Rotated resolution {width}x{height} exceeds Full HD")
            print(f"Applying downscale factor: {downscale_factor:.3f}")
        
        # Calculate final dimensions for processing
        processed_width = int(width * downscale_factor)
        processed_height = int(height * downscale_factor)
        
        # Swap back for actual processing if needed
        if rotation_angle in [90, 270]:
            processed_width, processed_height = processed_height, processed_width
        
        print(f"Final processing dimensions: {processed_width}x{processed_height}")
        return processed_width, processed_height
    
    def generate_camera_info(self, frames_folder, output_path):
        """
        Generate a cameras.txt file with extrinsic camera parameters for each frame.
        
        Args:
            frames_folder: Folder containing extracted frames
            output_path: Path to save the camera information
            
        Returns:
            Path to the camera information file
        """
        # Find all image files in the directory
        image_files = sorted(glob(os.path.join(frames_folder, "*.jpg")) + 
                           glob(os.path.join(frames_folder, "*.png")))
        
        if not image_files:
            print(f"No images found in {frames_folder}")
            return None
        
        # Create camera parameters
        with open(output_path, "w") as f:
            f.write("# Camera extrinsic parameters\n")
            f.write("# image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name\n")
            
            for idx, image_path in enumerate(image_files):
                image_id = idx + 1
                image_name = os.path.basename(image_path)
                camera_id = 1  # Using a single camera model for all frames
                
                # Calculate camera parameters based on frame position
                # We'll create a simple camera path - this should be customized for your specific needs
                qvec, tvec = self._calculate_camera_parameters_for_frame(idx, len(image_files))
                
                # Format: image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name
                camera_line = f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {camera_id} {image_name}\n"
                f.write(camera_line)
                
                # Add dummy point correspondences line (required by the format)
                dummy_point_line = "0\n"
                f.write(dummy_point_line)
        
        print(f"Camera information saved to {output_path}")
        return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from video with intelligent selection")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", required=True, help="Output folder for extracted frames")
    parser.add_argument("--max_frames", "-m", type=int, default=120, help="Maximum number of frames to extract")
    parser.add_argument("--downscale", "-d", type=float, default=1.0, help="Downscale factor (1.0 = original size)")
    parser.add_argument("--auto_fhd", "-a", action="store_true", help="Automatically downscale to Full HD if larger")
    parser.add_argument("--rotation", "-r", type=int, help="Force rotation angle (0, 90, 180, 270)")
    
    args = parser.parse_args()
    
    extractor = FrameExtractor()
    extractor.extract_frames(
        args.video,
        args.output,
        max_frames=args.max_frames,
        downscale_factor=args.downscale,
        auto_downscale_to_fhd=args.auto_fhd,
        force_rotation=args.rotation
    )