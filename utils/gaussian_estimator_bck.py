import cv2
import numpy as np
import os
import concurrent.futures
from glob import glob

class GaussianEstimator:
    """
    A class to estimate the optimal number of Gaussians for Gaussian Splatting
    based on image complexity analysis.
    """
    
    def estimate_optimal_gaussians(self, input_dir, sample_frames=10, use_gpu=False,
                                  base_gaussians=500000, complexity_factor=1000000, 
                                  depth_factor=1500000, min_gaussians=500000, 
                                  max_gaussians=5000000):
        """
        Estimate the optimal number of gaussians based on scene complexity
        by analyzing the input frames.
        
        Args:
            input_dir: Directory containing input frames
            sample_frames: Number of frames to analyze
            use_gpu: Whether to use GPU acceleration (not fully implemented)
            base_gaussians: Base number of gaussians for any scene
            complexity_factor: Scaling factor for visual complexity
            depth_factor: Scaling factor for depth complexity
            min_gaussians: Minimum number of gaussians to return
            max_gaussians: Maximum number of gaussians to return
            
        Returns:
            int: Estimated number of gaussians needed
        """
        # Find all image files in the directory
        image_files = sorted(glob(os.path.join(input_dir, "*.jpg")) + 
                            glob(os.path.join(input_dir, "*.png")))

        if not image_files:
            print("No images found in the input directory")
            return base_gaussians  # Default value if no images
        
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
        
        print(f"Analyzing {len(sample_images)} frames for complexity estimation")
        
        # Use multithreading to speed up image analysis
        complexity_scores = []
        depth_complexity_scores = []
        
        # Process images in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.analyze_image_complexity, sample_images))
        
        for complexity, depth in results:
            complexity_scores.append(complexity)
            depth_complexity_scores.append(depth)
        
        # Calculate average complexity
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        avg_depth_complexity = sum(depth_complexity_scores) / len(depth_complexity_scores) if depth_complexity_scores else 0
        
        # Combined estimate gives more weight to depth complexity for 3D scenes
        estimated_gaussians = int(base_gaussians + 
                                (avg_complexity * complexity_factor) + 
                                (avg_depth_complexity * depth_factor))
        
        # Limit the range to avoid extreme values
        estimated_gaussians = max(min_gaussians, min(estimated_gaussians, max_gaussians))
        
        print(f"Scene complexity analysis:")
        print(f"Visual complexity: {avg_complexity:.4f}")
        print(f"Depth complexity: {avg_depth_complexity:.4f}")
        print(f"Estimated gaussians needed: {estimated_gaussians}")
        
        return estimated_gaussians
    
    def analyze_image_complexity(self, img_path):
        """
        Analyze the complexity of an image using various metrics.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            tuple: (visual_complexity, depth_complexity)
        """
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            return 0, 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Calculate gradient (edges) as a measure of detail
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.sum(gradient_magnitude > 30) / (gray.shape[0] * gray.shape[1])
        
        # 2. Calculate entropy as a measure of texture complexity
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        non_zero_hist = hist[hist > 0]
        entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))
        
        # 3. Analyze spatial frequency
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        high_freq_content = np.mean(magnitude_spectrum > 150)
        
        # 4. Analyze depth complexity using edge gradients
        depth_score = np.std(gradient_magnitude) / 255.0
        
        # Combine the metrics for visual complexity
        combined_complexity = (edge_density * 0.4 + entropy * 0.3 + high_freq_content * 0.3)
        
        return combined_complexity, depth_score
    
    def estimate_from_pointcloud(self, point_cloud_path, density_factor=1.5, min_gaussians=300000, max_gaussians=3000000):
        """
        Estimate the optimal number of gaussians based on a point cloud.
        This is an alternative approach to the image-based estimation.
        
        Args:
            point_cloud_path: Path to the point cloud file (.ply or .pts)
            density_factor: Multiplication factor to account for densification
            min_gaussians: Minimum number of gaussians
            max_gaussians: Maximum number of gaussians
            
        Returns:
            int: Estimated number of gaussians
        """
        try:
            # Very simple approach: count the number of points and multiply by density factor
            # For a proper implementation, you'd need to parse the point cloud file format
            # and compute spatial statistics.
            
            # This is a placeholder for a real implementation
            points_count = self._count_points_in_pointcloud(point_cloud_path)
            
            if points_count == 0:
                print(f"Warning: Could not determine point count in {point_cloud_path}")
                return min_gaussians
                
            estimated_gaussians = int(points_count * density_factor)
            
            # Limit the range
            estimated_gaussians = max(min_gaussians, min(estimated_gaussians, max_gaussians))
            
            print(f"Point cloud analysis:")
            print(f"Number of points: {points_count}")
            print(f"Density factor: {density_factor}")
            print(f"Estimated gaussians needed: {estimated_gaussians}")
            
            return estimated_gaussians
            
        except Exception as e:
            print(f"Error analyzing point cloud: {e}")
            return min_gaussians
    
    def _count_points_in_pointcloud(self, point_cloud_path):
        """
        Count the number of points in a point cloud file.
        
        Args:
            point_cloud_path: Path to the point cloud file
            
        Returns:
            int: Number of points
        """
        # This is a simplified implementation that works for PLY ASCII files
        # A real implementation would need to handle different file formats
        try:
            if point_cloud_path.lower().endswith('.ply'):
                # For PLY files
                with open(point_cloud_path, 'r') as f:
                    header_end = False
                    vertex_count = 0
                    
                    for line in f:
                        line = line.strip()
                        if not header_end:
                            if line.startswith('element vertex'):
                                vertex_count = int(line.split()[2])
                            elif line == 'end_header':
                                header_end = True
                                return vertex_count
            
            # For other formats, you would need to implement specific parsers
            print(f"Unsupported point cloud format: {point_cloud_path}")
            return 0
            
        except Exception as e:
            print(f"Error reading point cloud file: {e}")
            return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Estimate optimal number of gaussians for Gaussian Splatting")
    parser.add_argument("--input", "-i", required=True, help="Path to input directory containing frames")
    parser.add_argument("--pointcloud", "-p", help="Path to point cloud file (alternative to image analysis)")
    parser.add_argument("--samples", "-s", type=int, default=10, help="Number of frames to sample for analysis")
    parser.add_argument("--base", "-b", type=int, default=500000, help="Base number of gaussians")
    parser.add_argument("--complexity_factor", "-c", type=int, default=1000000, help="Scaling factor for visual complexity")
    parser.add_argument("--depth_factor", "-d", type=int, default=1500000, help="Scaling factor for depth complexity")
    parser.add_argument("--min", "-min", type=int, default=500000, help="Minimum number of gaussians")
    parser.add_argument("--max", "-max", type=int, default=5000000, help="Maximum number of gaussians")
    
    args = parser.parse_args()
    
    estimator = GaussianEstimator()
    
    if args.pointcloud:
        # Estimate from point cloud if provided
        estimated = estimator.estimate_from_pointcloud(
            args.pointcloud,
            min_gaussians=args.min,
            max_gaussians=args.max
        )
    else:
        # Otherwise estimate from images
        estimated = estimator.estimate_optimal_gaussians(
            args.input,
            sample_frames=args.samples,
            base_gaussians=args.base,
            complexity_factor=args.complexity_factor,
            depth_factor=args.depth_factor,
            min_gaussians=args.min,
            max_gaussians=args.max
        )
    
    print(f"Final gaussian estimate: {estimated}")