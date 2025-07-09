import cv2
import numpy as np
import os
from collections import deque
import time
from pymediainfo import MediaInfo
import hashlib
import concurrent.futures
from glob import glob
from scipy.spatial.transform import Rotation
import math

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

    def calculate_sharp_frames_params(self,video_path, quality_level):
        """
        Calcola parametri ottimali per sharp-frames usando logica del FrameExtractor
        """
        frame_extractor = FrameExtractor()
        
        # 1. Ottieni info video + rotazione
        cap = cv2.VideoCapture(video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        cap.release()
        
        # 2. Usa le funzioni esistenti per rotazione
        rotation_angle = frame_extractor._get_normalized_rotation(video_path)
        
        # 3. Calcola dimensioni considerando rotazione
        width, height = original_width, original_height
        if rotation_angle in [90, 270]:
            width, height = height, width  # Swap per rotazione
        
        # 4. Calcola target width per sharp-frames
        target_width = None
        max_dimension = max(width, height)
        if max_dimension > 1920:  # Serve downscale
            scale_factor = 1920 / max_dimension
            target_width = int(width * scale_factor)
        
        # 5. Calcola fps basato su durata video e qualità
        target_frames_map = {
            'fast': 60,
            'balanced': 120,
            'quality': 180,
            'ultra': 240
        }
        
        target_frames = target_frames_map[quality_level]
        extraction_fps = target_frames / duration
        
        # Cap al fps originale del video
        extraction_fps = min(extraction_fps, video_fps)
        
        # Minimo assoluto per evitare errori
        extraction_fps = max(0.1, extraction_fps)
        
        return {
            'width': target_width,
            'fps': round(extraction_fps, 2)
        }

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
    
    
    

    def estimate_optimal_gaussians(self, input_dir, sample_frames=10, use_gpu=False):
        """
        Estimate the optimal number of gaussians based on scene complexity
        by analyzing the input frames.
        
        Args:
            input_dir: Directory containing input frames
            sample_frames: Number of frames to analyze
            
        Returns:
            int: Estimated number of gaussians needed
        """
        # Find all image files in the directory
        image_files = sorted(glob(os.path.join(input_dir, "*.jpg")) + 
                            glob(os.path.join(input_dir, "*.png")))

        if not image_files:
            print("No images found in the input directory")
            return 500000  # Default value if no images
        
        # Calculate the interval between images to select
        total_frames = len(image_files)
        interval = max(1, total_frames // sample_frames)  # Calculate interval between images
        
        # Select frames to analyze at equal distances
        sample_images = [image_files[i] for i in range(0, total_frames, interval)]
        
        # If there are fewer selected images than required, add the last images
        sample_images = sample_images[:sample_frames]
        while len(sample_images) < min(sample_frames, total_frames):
            idx = len(sample_images)
            if idx < total_frames:
                sample_images.append(image_files[idx])
        
        # Use multithreading to speed up image analysis
        complexity_scores = []
        depth_complexity_scores = []
        
        def analyze_image_complexity(img_path):
            # Upload image to GPU if needed
            img = cv2.imread(img_path)
            if img is None:
                return 0, 0
            
            
            # CPU-only version for the rest of the code
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Calculate gradient (edges) as a measure of detail
            # Misura quanti bordi/dettagli ci sono nell'immagine utilizzando
            #  l'operatore Sobel per calcolare i gradienti
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_density = np.sum(gradient_magnitude > 30) / (gray.shape[0] * gray.shape[1])
            
            # 2. Calculate entropy as a measure of texture complexity
            # misura l'imprevedibilità dell'immagine, che corrisponde alla complessità della texture
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / np.sum(hist)
            non_zero_hist = hist[hist > 0]
            entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))
            
            # 3. Analyze spatial frequency
            # Analizza lo spettro di frequenza dell'immagine 
            # usando la trasformata di Fourier per valutare la quantità di dettagli fini
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            high_freq_content = np.mean(magnitude_spectrum > 150)
            
            # 4. Analyze depth complexity using edge gradients
            # Stima la complessità della struttura 3D analizzando la variazione dei gradienti
            depth_score = np.std(gradient_magnitude) / 255.0
            
            # Combine the metrics
            combined_complexity = (edge_density * 0.4 + entropy * 0.3 + 
                                high_freq_content * 0.3)
            
            return combined_complexity, depth_score
        
        # Process images in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(analyze_image_complexity, sample_images))
        
        for complexity, depth in results:
            complexity_scores.append(complexity)
            depth_complexity_scores.append(depth)
        
        # Calculate average complexity
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        avg_depth_complexity = sum(depth_complexity_scores) / len(depth_complexity_scores) if depth_complexity_scores else 0
        
        # Map complexity to a range of gaussians
        base_gaussians = 500000
        complexity_factor = 1000000
        depth_factor = 1500000
        
        # Combined estimate gives more weight to depth complexity for 3D scenes
        estimated_gaussians = int(base_gaussians + 
                                (avg_complexity * complexity_factor) + 
                                (avg_depth_complexity * depth_factor))
        
        # Limit the range to avoid extreme values
        min_gaussians = 500000
        max_gaussians = 5000000
        estimated_gaussians = max(min_gaussians, min(estimated_gaussians, max_gaussians))
        
        print(f"Scene complexity analysis:")
        print(f"Visual complexity: {avg_complexity:.4f}")
        print(f"Depth complexity: {avg_depth_complexity:.4f}")
        print(f"Estimated gaussians needed: {estimated_gaussians}")
        
        return estimated_gaussians