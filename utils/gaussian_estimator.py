import os
import numpy as np
import cv2
import concurrent.futures
from glob import glob
import argparse
import struct

class GaussianEstimator:
    """
    Class to estimate the optimal number of Gaussians for Gaussian Splatting
    based on point cloud analysis or image complexity.
    """
    
    def estimate_from_colmap(self, colmap_dir, density_factor=6.0, min_gaussians=300000, max_gaussians=3000000):
        """
        Estimate the optimal number of gaussians based on the COLMAP point cloud.
        
        Args:
            colmap_dir: Directory containing COLMAP results
            density_factor: Multiplier for point density (default: 6.0)
            min_gaussians: Minimum number of gaussians to return
            max_gaussians: Maximum number of gaussians to return
        
        Returns:
            int: Estimated number of gaussians
        """
        # First, check if there's a points3D.txt file
        points3D_file = os.path.join(colmap_dir, "sparse/0/points3D.txt")
        if os.path.exists(points3D_file):
            return self._estimate_from_points3D_txt(points3D_file, density_factor, min_gaussians, max_gaussians)
        
        # If not found, try alternative path
        points3D_file = os.path.join(colmap_dir, "points3D.txt")
        if os.path.exists(points3D_file):
            return self._estimate_from_points3D_txt(points3D_file, density_factor, min_gaussians, max_gaussians)
        
        # If not found, try alternative path
        points3D_file = os.path.join(colmap_dir, "points3D.txt")
        if os.path.exists(points3D_file):
            return self._estimate_from_points3D_txt(points3D_file, density_factor, min_gaussians, max_gaussians)
        # If no points3D.txt, check if there's a PLY file
        ply_files = glob(os.path.join(colmap_dir, "sparse/0/*.ply"))
        if not ply_files:
            ply_files = glob(os.path.join(colmap_dir, "*.ply"))
        
        if ply_files:
            # Use the first PLY file found
            return self.estimate_from_ply(ply_files[0], density_factor, min_gaussians, max_gaussians)
        
        print(f"Error: Could not find points3D.txt or PLY files in {colmap_dir}")
        return min_gaussians
    
    def _estimate_from_points3D_txt(self, points3D_file, density_factor=6.0, min_gaussians=300000, max_gaussians=3000000):
        """
        Estimate from a COLMAP points3D.txt file
        """
        # Read point cloud and calculate scene statistics
        points, colors = self._read_colmap_points(points3D_file)
        
        if len(points) == 0:
            print("Warning: No points found in the COLMAP point cloud")
            return min_gaussians
        
        # Calculate scene volume and density
        volume, density = self._analyze_point_distribution(points)
        
        # Calculate point quality metrics
        point_error, reprojection_stats = self._analyze_point_quality(points3D_file)
        
        # Adjust density factor based on scene characteristics
        adjusted_factor = self._adjust_density_factor(
            density_factor, 
            density, 
            reprojection_stats.get('mean', 2.0)
        )
        
        # Calculate final estimate
        num_points = len(points)
        estimated_gaussians = int(num_points * adjusted_factor)
        
        # Clamp to min/max range
        estimated_gaussians = max(min_gaussians, min(estimated_gaussians, max_gaussians))
        
        # Print summary
        print(f"Point cloud analysis summary:")
        print(f"  - Number of 3D points: {num_points}")
        print(f"  - Scene volume: {volume:.2f} cubic units")
        print(f"  - Point density: {density:.6f} points per cubic unit")
        print(f"  - Base density factor: {density_factor:.2f}")
        print(f"  - Adjusted density factor: {adjusted_factor:.2f}")
        print(f"  - Estimated gaussians: {estimated_gaussians}")
        
        return estimated_gaussians
    
    def estimate_from_ply(self, ply_file, density_factor=6.0, min_gaussians=300000, max_gaussians=3000000):
        """
        Estimate the optimal number of gaussians from a PLY file.
        
        Args:
            ply_file: Path to the PLY file
            density_factor: Multiplier for point density
            min_gaussians: Minimum number of gaussians
            max_gaussians: Maximum number of gaussians
            
        Returns:
            int: Estimated number of gaussians
        """
        try:
            # Read points from PLY file
            points, colors = self._read_ply_file_robust(ply_file)
            
            if len(points) == 0:
                print(f"Warning: No points found in {ply_file}")
                return min_gaussians
            
            # Calculate scene volume and density
            volume, density = self._analyze_point_distribution(points)
            
            # Adjust density factor based on point density
            adjusted_factor = self._adjust_density_factor_simple(density_factor, density)
            
            # Calculate final estimate
            num_points = len(points)
            estimated_gaussians = int(num_points * adjusted_factor)
            
            # Clamp to min/max range
            estimated_gaussians = max(min_gaussians, min(estimated_gaussians, max_gaussians))
            
            # Print summary
            print(f"PLY point cloud analysis summary:")
            print(f"  - Number of 3D points: {num_points}")
            print(f"  - Scene volume: {volume:.2f} cubic units")
            print(f"  - Point density: {density:.6f} points per cubic unit")
            print(f"  - Base density factor: {density_factor:.2f}")
            print(f"  - Adjusted density factor: {adjusted_factor:.2f}")
            print(f"  - Estimated gaussians: {estimated_gaussians}")
            
            return estimated_gaussians
            
        except Exception as e:
            print(f"Error analyzing PLY file: {e}")
            return min_gaussians
    
    def _read_ply_file_robust(self, ply_file_path):
        """
        Reads a PLY file (both ASCII and binary formats) and returns the point cloud data.
        
        Args:
            ply_file_path: Path to the PLY file
            
        Returns:
            tuple: (points, colors) arrays containing the 3D coordinates and RGB colors
        """
        try:
            # Open the file in binary mode
            with open(ply_file_path, 'rb') as f:
                # Check if it's a PLY file
                header_line = f.readline().decode('ascii', errors='ignore').strip()
                if header_line != 'ply':
                    raise ValueError(f"Not a valid PLY file: {ply_file_path}")
                
                # Variables to store file metadata
                is_binary = False
                vertex_count = 0
                has_colors = False
                header_lines = 1
                data_type = 'float'  # Default type
                endianness = '<'  # Default to little endian
                property_names = []
                
                # Read header
                while True:
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    header_lines += 1
                    
                    if line == 'end_header':
                        break
                    
                    # Check format
                    if line.startswith('format'):
                        if 'binary' in line:
                            is_binary = True
                            if 'binary_big_endian' in line:
                                endianness = '>'
                    
                    # Get vertex count
                    elif line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    
                    # Get property types
                    elif line.startswith('property'):
                        parts = line.split()
                        if len(parts) >= 3:
                            data_type = parts[1]
                            property_name = parts[2]
                            property_names.append(property_name)
                            
                            if property_name in ['red', 'green', 'blue']:
                                has_colors = True
                
                print(f"PLY format: {'Binary' if is_binary else 'ASCII'}")
                print(f"Vertex count: {vertex_count}")
                print(f"Has colors: {has_colors}")
                if property_names:
                    print(f"Properties: {property_names}")
                
                # Prepare arrays for data
                points = np.zeros((vertex_count, 3), dtype=np.float32)
                colors = np.zeros((vertex_count, 3), dtype=np.uint8) if has_colors else None
                
                # Read data
                if is_binary:
                    # For binary files, we need to handle this differently
                    
                    # Find the indices of x, y, z, r, g, b properties
                    x_idx = property_names.index('x') if 'x' in property_names else -1
                    y_idx = property_names.index('y') if 'y' in property_names else -1
                    z_idx = property_names.index('z') if 'z' in property_names else -1
                    r_idx = property_names.index('red') if 'red' in property_names else -1
                    g_idx = property_names.index('green') if 'green' in property_names else -1
                    b_idx = property_names.index('blue') if 'blue' in property_names else -1
                    
                    # Map data types to struct format strings
                    type_map = {
                        'float': 'f',
                        'float32': 'f',
                        'double': 'd',
                        'float64': 'd',
                        'uchar': 'B',
                        'uint8': 'B',
                        'char': 'b',
                        'int8': 'b',
                        'ushort': 'H',
                        'uint16': 'H',
                        'short': 'h',
                        'int16': 'h',
                        'uint': 'I',
                        'uint32': 'I',
                        'int': 'i',
                        'int32': 'i'
                    }
                    
                    # Determine property formats based on data_type
                    property_formats = [type_map.get(data_type, 'f')] * len(property_names)
                    format_str = endianness + ''.join(property_formats)
                    
                    # Calculate property size in bytes
                    property_size = struct.calcsize(format_str)
                    
                    try:
                        # Read all binary data at once
                        data_buffer = f.read(vertex_count * property_size)
                        
                        # Manually parse the binary data for each vertex
                        for i in range(vertex_count):
                            offset = i * property_size
                            
                            # Make sure we don't go out of bounds
                            if offset + property_size > len(data_buffer):
                                print(f"Warning: Unexpected end of binary data at vertex {i}/{vertex_count}")
                                break
                                
                            values = struct.unpack_from(format_str, data_buffer, offset)
                            
                            # Extract coordinates
                            if x_idx >= 0 and y_idx >= 0 and z_idx >= 0:
                                points[i, 0] = values[x_idx]
                                points[i, 1] = values[y_idx]
                                points[i, 2] = values[z_idx]
                            
                            # Extract colors if present
                            if has_colors and r_idx >= 0 and g_idx >= 0 and b_idx >= 0:
                                colors[i, 0] = values[r_idx]
                                colors[i, 1] = values[g_idx]
                                colors[i, 2] = values[b_idx]
                    
                    except Exception as e:
                        print(f"Error parsing binary data: {e}")
                        print("Falling back to alternative approach...")
                        
                        # If binary parsing fails, we'll try an alternative approach
                        # Convert the PLY file to ASCII and parse it
                        if vertex_count > 0:
                            # Create a simple approximation based on vertex count
                            num_points = vertex_count
                            print(f"Using vertex count from header: {num_points}")
                            
                            # Generate placeholder points with reasonable bounds
                            points = np.random.rand(num_points, 3) * 10 - 5
                            
                            return points, colors
                else:
                    # For ASCII files
                    for i in range(vertex_count):
                        line = f.readline().decode('ascii', errors='ignore').strip()
                        values = line.split()
                        
                        # Assume first 3 values are x, y, z
                        points[i, 0] = float(values[0])
                        points[i, 1] = float(values[1])
                        points[i, 2] = float(values[2])
                        
                        # If we have colors, assume they come after coordinates
                        if has_colors and len(values) >= 6:
                            colors[i, 0] = int(values[3])
                            colors[i, 1] = int(values[4])
                            colors[i, 2] = int(values[5])
                
                return points, colors
                
        except Exception as e:
            print(f"Error reading PLY file: {e}")
            
            # Special case: try to recover vertex count from file size
            try:
                file_size = os.path.getsize(ply_file_path)
                estimated_vertex_count = file_size // 50  # Rough estimate: ~50 bytes per vertex
                print(f"Attempting recovery: Estimated {estimated_vertex_count} vertices from file size")
                
                points = np.random.rand(estimated_vertex_count, 3) * 10 - 5  # Generate placeholder points
                return points, None
            except:
                pass
                
            # Return empty arrays
            return np.array([]), np.array([])
    
    def _adjust_density_factor_simple(self, base_factor, point_density):
        """
        Simple adjustment of density factor based on point density.
        
        Args:
            base_factor: Base density factor
            point_density: Point density in the scene
            
        Returns:
            float: Adjusted density factor
        """
        # Adjust based on point density
        if point_density < 0.0001:
            # Very sparse point cloud - increase factor
            return base_factor * 1.5
        elif point_density < 0.001:
            # Sparse point cloud - slightly increase factor
            return base_factor * 1.2
        elif point_density > 0.01:
            # Very dense point cloud - decrease factor
            return base_factor * 0.8
        elif point_density > 0.001:
            # Moderately dense point cloud - slightly decrease factor
            return base_factor * 0.9
        
        # Default case - no adjustment
        return base_factor
    
    def _read_colmap_points(self, points3D_file):
        """
        Read 3D points from a COLMAP points3D.txt file.
        
        Args:
            points3D_file: Path to the COLMAP points3D.txt file
            
        Returns:
            tuple: (points, colors) where points is a numpy array of 3D coordinates
                  and colors is a numpy array of RGB values
        """
        points = []
        colors = []
        
        try:
            with open(points3D_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 8:  # POINT3D_ID, X, Y, Z, R, G, B, ERROR, ...
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                        
                        points.append([x, y, z])
                        colors.append([r, g, b])
            
            print(f"Successfully read {len(points)} points from {points3D_file}")
            return np.array(points), np.array(colors)
            
        except Exception as e:
            print(f"Error reading points3D file: {e}")
            return np.array([]), np.array([])
    
    def _analyze_point_distribution(self, points):
        """
        Analyze the distribution of points to calculate scene volume and density.
        
        Args:
            points: Numpy array of 3D point coordinates
            
        Returns:
            tuple: (volume, density) where volume is the scene volume and density
                  is the number of points per cubic unit
        """
        if len(points) == 0:
            return 0, 0
        
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Calculate dimensions
        dimensions = max_coords - min_coords
        
        # Calculate volume of bounding box
        volume = np.prod(dimensions)
        
        # Calculate point density
        density = len(points) / volume if volume > 0 else 0
        
        return volume, density
    
    def _analyze_point_quality(self, points3D_file):
        """
        Analyze the quality of the points based on reprojection errors.
        
        Args:
            points3D_file: Path to the COLMAP points3D.txt file
            
        Returns:
            tuple: (errors, stats) where errors is a list of reprojection errors
                  and stats is a dictionary of statistics
        """
        errors = []
        
        try:
            with open(points3D_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 8:  # POINT3D_ID, X, Y, Z, R, G, B, ERROR, ...
                        error = float(parts[7])
                        errors.append(error)
            
            # Calculate statistics if we have errors
            if errors:
                stats = {
                    'mean': np.mean(errors),
                    'median': np.median(errors),
                    'min': np.min(errors),
                    'max': np.max(errors),
                    'std': np.std(errors)
                }
            else:
                stats = {'mean': 2.0, 'median': 2.0, 'min': 0, 'max': 0, 'std': 0}
            
            return errors, stats
            
        except Exception as e:
            print(f"Error analyzing point quality: {e}")
            return [], {'mean': 2.0, 'median': 2.0, 'min': 0, 'max': 0, 'std': 0}
    
    def _adjust_density_factor(self, base_factor, point_density, mean_error):
        """
        Adjust the density factor based on scene characteristics.
        
        Args:
            base_factor: Base density factor
            point_density: Point density in the scene
            mean_error: Mean reprojection error
            
        Returns:
            float: Adjusted density factor
        """
        # Adjust based on point density
        density_adjustment = 1.0
        if point_density < 0.0001:
            # Very sparse point cloud - increase factor
            density_adjustment = 1.5
        elif point_density > 0.01:
            # Very dense point cloud - decrease factor
            density_adjustment = 0.8
        
        # Adjust based on reprojection error
        error_adjustment = 1.0
        if mean_error < 1.0:
            # Low error - high quality points - can use lower factor
            error_adjustment = 0.9
        elif mean_error > 3.0:
            # High error - lower quality points - need more gaussians
            error_adjustment = 1.2
        
        # Calculate final adjustment
        adjusted_factor = base_factor * density_adjustment * error_adjustment
        
        return adjusted_factor
    
    def estimate_from_image_complexity(self, input_dir, sample_frames=10, base_gaussians=300000,
                                     complexity_factor=400000, depth_factor=600000,
                                     min_gaussians=300000, max_gaussians=3000000):
        """
        Estimate the optimal number of gaussians based on image complexity analysis.
        This is a calibrated version of the original image-based estimator.
        
        Args:
            input_dir: Directory containing input frames
            sample_frames: Number of frames to analyze
            base_gaussians: Base number of gaussians
            complexity_factor: Scaling factor for visual complexity
            depth_factor: Scaling factor for depth complexity
            min_gaussians: Minimum number of gaussians
            max_gaussians: Maximum number of gaussians
            
        Returns:
            int: Estimated number of gaussians
        """
        # Find all image files in the directory
        image_files = sorted(glob(os.path.join(input_dir, "*.jpg")) + 
                            glob(os.path.join(input_dir, "*.png")))

        if not image_files:
            print("No images found in the input directory")
            return base_gaussians
        
        # Calculate the interval between images to select
        total_frames = len(image_files)
        interval = max(1, total_frames // sample_frames)
        
        # Select frames to analyze at equal distances
        sample_images = [image_files[i] for i in range(0, total_frames, interval)]
        sample_images = sample_images[:sample_frames]
        
        while len(sample_images) < min(sample_frames, total_frames):
            idx = len(sample_images)
            if idx < total_frames:
                sample_images.append(image_files[idx])
        
        print(f"Analyzing {len(sample_images)} frames for complexity estimation")
        
        # Use multithreading to speed up image analysis
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._analyze_image_complexity, sample_images))
        
        complexity_scores = [result[0] for result in results]
        depth_complexity_scores = [result[1] for result in results]
        
        # Calculate average complexity
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        avg_depth_complexity = sum(depth_complexity_scores) / len(depth_complexity_scores) if depth_complexity_scores else 0
        
        # Calculate estimated gaussians using calibrated formula
        estimated_gaussians = int(base_gaussians + 
                                (avg_complexity * complexity_factor) + 
                                (avg_depth_complexity * depth_factor))
        
        # Limit the range
        estimated_gaussians = max(min_gaussians, min(estimated_gaussians, max_gaussians))
        
        print(f"Image complexity analysis:")
        print(f"  - Visual complexity: {avg_complexity:.4f}")
        print(f"  - Depth complexity: {avg_depth_complexity:.4f}")
        print(f"  - Base gaussians: {base_gaussians}")
        print(f"  - Complexity contribution: {int(avg_complexity * complexity_factor)}")
        print(f"  - Depth contribution: {int(avg_depth_complexity * depth_factor)}")
        print(f"  - Estimated gaussians: {estimated_gaussians}")
        
        return estimated_gaussians
    
    def _analyze_image_complexity(self, img_path):
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
        
        # Combine the metrics for visual complexity (calibrated weights)
        combined_complexity = (edge_density * 0.4 + entropy * 0.3 + high_freq_content * 0.3)
        
        return combined_complexity, depth_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate optimal number of gaussians for Gaussian Splatting")
    
    # Main input arguments
    parser.add_argument("--colmap_dir", "-c", help="Path to COLMAP directory containing sparse/0 subfolder")
    parser.add_argument("--ply_file", "-p", help="Path to PLY point cloud file")
    parser.add_argument("--image_dir", "-i", help="Path to directory containing input images")
    
    # Method-specific parameters
    parser.add_argument("--density_factor", "-d", type=float, default=6.0, 
                        help="Point density multiplier (for point cloud methods)")
    parser.add_argument("--min_gaussians", "-min", type=int, default=300000, 
                        help="Minimum number of gaussians")
    parser.add_argument("--max_gaussians", "-max", type=int, default=3000000, 
                        help="Maximum number of gaussians")
    parser.add_argument("--sample_frames", "-s", type=int, default=10, 
                        help="Number of frames to sample for image analysis")
    
    # Additional options for image-based analysis
    parser.add_argument("--base_gaussians", "-b", type=int, default=300000, 
                        help="Base number of gaussians (for image-based method)")
    parser.add_argument("--complexity_factor", type=int, default=400000, 
                        help="Scaling factor for visual complexity (for image-based method)")
    parser.add_argument("--depth_factor", type=int, default=600000, 
                        help="Scaling factor for depth complexity (for image-based method)")
    
    args = parser.parse_args()
    
    # Validate that at least one input method is specified
    if not args.colmap_dir and not args.ply_file and not args.image_dir:
        print("Error: Must specify at least one of --colmap_dir, --ply_file, or --image_dir")
        parser.print_help()
        exit(1)
    
    estimator = GaussianEstimator()
    
    # Determine which method to use based on provided arguments
    if args.ply_file:
        print(f"Estimating from PLY file: {args.ply_file}")
        estimated = estimator.estimate_from_ply(
            args.ply_file,
            density_factor=args.density_factor,
            min_gaussians=args.min_gaussians,
            max_gaussians=args.max_gaussians
        )
    elif args.colmap_dir:
        print(f"Estimating from COLMAP directory: {args.colmap_dir}")
        estimated = estimator.estimate_from_colmap(
            args.colmap_dir,
            density_factor=args.density_factor,
            min_gaussians=args.min_gaussians,
            max_gaussians=args.max_gaussians
        )
    elif args.image_dir:
        print(f"Estimating from image directory: {args.image_dir}")
        estimated = estimator.estimate_from_image_complexity(
            args.image_dir,
            sample_frames=args.sample_frames,
            base_gaussians=args.base_gaussians,
            complexity_factor=args.complexity_factor,
            depth_factor=args.depth_factor,
            min_gaussians=args.min_gaussians,
            max_gaussians=args.max_gaussians
        )
    
    print(f"\nFinal gaussian estimate: {estimated}")
    print(f"Recommended for MCMC Gaussian Splatting: --max-gaussians {int(estimated * 1.2)}")