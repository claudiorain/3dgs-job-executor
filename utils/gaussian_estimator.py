import os
import numpy as np
import cv2
import concurrent.futures
from glob import glob
import argparse
import struct

class GaussianEstimator:
    """
    Enhanced class to estimate the optimal number of Gaussians for Gaussian Splatting
    with improved accuracy and less conservative estimates.
    """
    
    def estimate_from_colmap(self, colmap_dir, density_factor=8.0, min_gaussians=500000, max_gaussians=5000000):
        """
        Estimate the optimal number of gaussians based on the COLMAP point cloud.
        Updated with more aggressive defaults and improved analysis.
        
        Args:
            colmap_dir: Directory containing COLMAP results
            density_factor: Multiplier for point density (increased from 6.0 to 8.0)
            min_gaussians: Minimum number of gaussians (increased from 300k to 500k)
            max_gaussians: Maximum number of gaussians (increased from 3M to 5M)
        
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
        
        # If no points3D.txt, check if there's a PLY file
        ply_files = glob(os.path.join(colmap_dir, "sparse/0/*.ply"))
        if not ply_files:
            ply_files = glob(os.path.join(colmap_dir, "*.ply"))
        
        if ply_files:
            # Use the first PLY file found
            return self.estimate_from_ply(ply_files[0], density_factor, min_gaussians, max_gaussians)
        
        return min_gaussians
    
    def _estimate_from_points3D_txt(self, points3D_file, density_factor=8.0, min_gaussians=500000, max_gaussians=5000000):
        """
        Enhanced estimation from a COLMAP points3D.txt file with better scene analysis.
        """
        # Read point cloud and calculate scene statistics
        points, colors = self._read_colmap_points(points3D_file)
        
        if len(points) == 0:
            print("Warning: No points found in the COLMAP point cloud")
            return min_gaussians
        
        # Enhanced scene analysis
        volume, density, surface_area = self._analyze_scene_geometry(points)
        
        # Calculate point quality metrics
        point_error, reprojection_stats = self._analyze_point_quality(points3D_file)
        
        # Enhanced complexity analysis
        complexity_metrics = self._analyze_scene_complexity(points, colors)
        
        # Multi-factor adjustment of density factor
        adjusted_factor = self._calculate_adaptive_density_factor(
            density_factor, 
            density, 
            reprojection_stats.get('mean', 2.0),
            complexity_metrics,
            surface_area,
            len(points)
        )
        
        # Calculate final estimate with multiple methods and take the maximum
        num_points = len(points)
        
        # Method 1: Point-based scaling
        estimate_1 = int(num_points * adjusted_factor)
        
        # Method 2: Surface area based estimation
        estimate_2 = int(surface_area * 50000)  # ~50k gaussians per unit surface area
        
        # Method 3: Volume-based estimation with complexity weighting
        volume_density_target = 100000  # Target gaussians per cubic unit
        estimate_3 = int(volume * volume_density_target * complexity_metrics['overall_complexity'])
        
        # Take the maximum of the three methods (less conservative)
        estimated_gaussians = max(estimate_1, estimate_2, estimate_3)
        
        # Apply bounds
        estimated_gaussians = max(min_gaussians, min(estimated_gaussians, max_gaussians))
        
        # Print enhanced summary
        print(f"Enhanced point cloud analysis summary:")
        print(f"  - Number of 3D points: {num_points}")
        print(f"  - Scene volume: {volume:.2f} cubic units")
        print(f"  - Scene surface area: {surface_area:.2f} square units")
        print(f"  - Point density: {density:.6f} points per cubic unit")
        print(f"  - Scene complexity: {complexity_metrics['overall_complexity']:.3f}")
        print(f"  - Base density factor: {density_factor:.2f}")
        print(f"  - Adjusted density factor: {adjusted_factor:.2f}")
        print(f"  - Estimate method 1 (point-based): {estimate_1}")
        print(f"  - Estimate method 2 (surface-based): {estimate_2}")
        print(f"  - Estimate method 3 (volume-based): {estimate_3}")
        print(f"  - Final estimate (max): {estimated_gaussians}")
        
        return estimated_gaussians
    
    def _analyze_scene_geometry(self, points):
        """
        Enhanced geometric analysis including surface area estimation.
        
        Args:
            points: Numpy array of 3D point coordinates
            
        Returns:
            tuple: (volume, density, surface_area)
        """
        if len(points) == 0:
            return 0, 0, 0
        
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords
        
        # Calculate volume of bounding box
        volume = np.prod(dimensions)
        
        # Calculate point density
        density = len(points) / volume if volume > 0 else 0
        
        # Estimate surface area using convex hull approximation
        try:
            from scipy.spatial import ConvexHull
            if len(points) > 4:  # Need at least 4 points for 3D hull
                hull = ConvexHull(points)
                surface_area = hull.area
            else:
                # Fallback: estimate surface area from bounding box
                surface_area = 2 * (dimensions[0] * dimensions[1] + 
                                   dimensions[1] * dimensions[2] + 
                                   dimensions[0] * dimensions[2])
        except ImportError:
            # Fallback if scipy is not available
            surface_area = 2 * (dimensions[0] * dimensions[1] + 
                               dimensions[1] * dimensions[2] + 
                               dimensions[0] * dimensions[2])
        except Exception:
            # Another fallback
            surface_area = np.sqrt(volume) * 6  # Rough approximation
        
        return volume, density, surface_area
    
    def _analyze_scene_complexity(self, points, colors):
        """
        Analyze scene complexity based on point distribution and color variation.
        
        Args:
            points: Numpy array of 3D point coordinates
            colors: Numpy array of RGB colors
            
        Returns:
            dict: Dictionary of complexity metrics
        """
        metrics = {}
        
        if len(points) == 0:
            return {'overall_complexity': 1.0}
        
        # 1. Spatial distribution complexity
        # Calculate nearest neighbor distances
        from scipy.spatial.distance import pdist
        if len(points) > 1:
            distances = pdist(points)
            spatial_std = np.std(distances)
            spatial_mean = np.mean(distances)
            spatial_complexity = min(2.0, spatial_std / spatial_mean) if spatial_mean > 0 else 1.0
        else:
            spatial_complexity = 1.0
        
        # 2. Color variation complexity
        if colors is not None and len(colors) > 0:
            color_std = np.std(colors.astype(float), axis=0)
            color_complexity = np.mean(color_std) / 255.0
        else:
            color_complexity = 0.5  # Default moderate complexity
        
        # 3. Point density variation
        # Divide space into voxels and analyze density variation
        n_voxels = min(50, int(len(points) ** (1/3)))  # Adaptive voxel count
        if n_voxels > 1:
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            voxel_size = (max_coords - min_coords) / n_voxels
            
            voxel_counts = []
            for i in range(n_voxels):
                for j in range(n_voxels):
                    for k in range(n_voxels):
                        voxel_min = min_coords + np.array([i, j, k]) * voxel_size
                        voxel_max = voxel_min + voxel_size
                        
                        in_voxel = np.all((points >= voxel_min) & (points < voxel_max), axis=1)
                        voxel_counts.append(np.sum(in_voxel))
            
            density_variation = np.std(voxel_counts) / (np.mean(voxel_counts) + 1e-6)
        else:
            density_variation = 1.0
        
        # Combine metrics
        metrics['spatial_complexity'] = spatial_complexity
        metrics['color_complexity'] = color_complexity
        metrics['density_variation'] = min(2.0, density_variation)
        metrics['overall_complexity'] = np.mean([
            spatial_complexity * 0.4,
            color_complexity * 0.3 + 0.5,  # Boost color importance
            metrics['density_variation'] * 0.3
        ])
        
        # Ensure minimum complexity of 1.0 for realistic scenes
        metrics['overall_complexity'] = max(1.0, metrics['overall_complexity'])
        
        return metrics
    
    def _calculate_adaptive_density_factor(self, base_factor, point_density, mean_error, 
                                         complexity_metrics, surface_area, num_points):
        """
        Enhanced adaptive density factor calculation with multiple considerations.
        
        Args:
            base_factor: Base density factor
            point_density: Point density in the scene
            mean_error: Mean reprojection error
            complexity_metrics: Dictionary of complexity metrics
            surface_area: Estimated surface area
            num_points: Number of points in the cloud
            
        Returns:
            float: Adjusted density factor
        """
        # Start with base factor
        adjusted_factor = base_factor
        
        # 1. Point density adjustment (more aggressive)
        if point_density < 0.0001:
            adjusted_factor *= 2.0  # Very sparse - double the factor
        elif point_density < 0.001:
            adjusted_factor *= 1.5  # Sparse - increase significantly
        elif point_density > 0.01:
            adjusted_factor *= 1.1  # Dense - only slight increase (less conservative)
        elif point_density > 0.001:
            adjusted_factor *= 1.2  # Moderate - still increase
        
        # 2. Error-based adjustment (more nuanced)
        if mean_error < 0.5:
            adjusted_factor *= 1.1  # Very good quality - can afford more gaussians
        elif mean_error < 1.0:
            adjusted_factor *= 1.05  # Good quality - slight increase
        elif mean_error > 5.0:
            adjusted_factor *= 1.4  # Poor quality - need many more gaussians
        elif mean_error > 3.0:
            adjusted_factor *= 1.3  # Moderate quality - increase more
        
        # 3. Complexity-based adjustment
        complexity_factor = complexity_metrics.get('overall_complexity', 1.0)
        adjusted_factor *= (0.8 + 0.4 * complexity_factor)  # Scale with complexity
        
        # 4. Scale-based adjustment
        if num_points < 50000:
            adjusted_factor *= 1.3  # Small scenes need relatively more gaussians
        elif num_points > 500000:
            adjusted_factor *= 0.95  # Very large scenes can be slightly more efficient
        
        # 5. Surface area consideration
        points_per_surface_unit = num_points / (surface_area + 1e-6)
        if points_per_surface_unit < 1000:
            adjusted_factor *= 1.2  # Sparse surface coverage
        
        return adjusted_factor
    
    def estimate_from_ply(self, ply_file, density_factor=8.0, min_gaussians=500000, max_gaussians=5000000):
        """
        Enhanced PLY-based estimation with updated defaults.
        """
        try:
            # Read points from PLY file
            points, colors = self._read_ply_file_robust(ply_file)
            
            if len(points) == 0:
                print(f"Warning: No points found in {ply_file}")
                return min_gaussians
            
            # Enhanced analysis
            volume, density, surface_area = self._analyze_scene_geometry(points)
            complexity_metrics = self._analyze_scene_complexity(points, colors)
            
            # Multi-method estimation
            adjusted_factor = self._adjust_density_factor_enhanced(
                density_factor, density, complexity_metrics, surface_area, len(points)
            )
            
            # Multiple estimation methods
            num_points = len(points)
            estimate_1 = int(num_points * adjusted_factor)
            estimate_2 = int(surface_area * 50000)
            estimate_3 = int(volume * 100000 * complexity_metrics['overall_complexity'])
            
            # Take maximum (less conservative)
            estimated_gaussians = max(estimate_1, estimate_2, estimate_3)
            estimated_gaussians = max(min_gaussians, min(estimated_gaussians, max_gaussians))
            
            # Enhanced summary
            print(f"Enhanced PLY analysis summary:")
            print(f"  - Number of 3D points: {num_points}")
            print(f"  - Scene volume: {volume:.2f} cubic units")
            print(f"  - Scene surface area: {surface_area:.2f} square units")
            print(f"  - Point density: {density:.6f} points per cubic unit")
            print(f"  - Scene complexity: {complexity_metrics['overall_complexity']:.3f}")
            print(f"  - Adjusted density factor: {adjusted_factor:.2f}")
            print(f"  - Multi-method estimates: {estimate_1}, {estimate_2}, {estimate_3}")
            print(f"  - Final estimate: {estimated_gaussians}")
            
            return estimated_gaussians
            
        except Exception as e:
            print(f"Error analyzing PLY file: {e}")
            return min_gaussians
    
    def _adjust_density_factor_enhanced(self, base_factor, point_density, complexity_metrics, surface_area, num_points):
        """
        Enhanced density factor adjustment with multiple considerations.
        """
        adjusted_factor = base_factor
        
        # Density adjustment (more aggressive)
        if point_density < 0.0001:
            adjusted_factor *= 2.0
        elif point_density < 0.001:
            adjusted_factor *= 1.6
        elif point_density > 0.01:
            adjusted_factor *= 1.1
        else:
            adjusted_factor *= 1.2
        
        # Complexity adjustment
        complexity = complexity_metrics.get('overall_complexity', 1.0)
        adjusted_factor *= (0.8 + 0.5 * complexity)
        
        # Scale adjustment
        if num_points < 50000:
            adjusted_factor *= 1.3
        elif num_points > 500000:
            adjusted_factor *= 0.95
        
        return adjusted_factor
    
    def estimate_from_image_complexity(self, input_dir, sample_frames=15, base_gaussians=500000,
                                     complexity_factor=600000, depth_factor=800000,
                                     min_gaussians=500000, max_gaussians=5000000):
        """
        Enhanced image-based estimation with more aggressive parameters and better analysis.
        """
        # Find all image files
        image_files = sorted(glob(os.path.join(input_dir, "*.jpg")) + 
                            glob(os.path.join(input_dir, "*.png")))

        if not image_files:
            print("No images found in the input directory")
            return base_gaussians
        
        # Improved sampling strategy
        total_frames = len(image_files)
        if total_frames <= sample_frames:
            sample_images = image_files
        else:
            # Sample from beginning, middle, and end for better coverage
            indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
            sample_images = [image_files[i] for i in indices]
        
        print(f"Analyzing {len(sample_images)} frames for enhanced complexity estimation")
        
        # Multi-threaded analysis
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._analyze_image_complexity_enhanced, sample_images))
        
        # Extract and analyze results
        complexity_scores = [r[0] for r in results]
        depth_scores = [r[1] for r in results]
        detail_scores = [r[2] for r in results]
        motion_scores = [r[3] for r in results]
        
        # Calculate enhanced metrics
        avg_complexity = np.mean(complexity_scores)
        avg_depth = np.mean(depth_scores)
        avg_detail = np.mean(detail_scores)
        avg_motion = np.mean(motion_scores)
        
        # Percentile-based analysis for outliers
        high_complexity_ratio = np.sum(np.array(complexity_scores) > np.percentile(complexity_scores, 75)) / len(complexity_scores)
        
        # Enhanced estimation formula
        base_estimate = base_gaussians
        complexity_contribution = avg_complexity * complexity_factor
        depth_contribution = avg_depth * depth_factor
        detail_contribution = avg_detail * 400000  # New detail factor
        motion_contribution = avg_motion * 300000   # New motion factor
        outlier_boost = high_complexity_ratio * 200000  # Boost for complex scenes
        
        estimated_gaussians = int(base_estimate + complexity_contribution + 
                                depth_contribution + detail_contribution + 
                                motion_contribution + outlier_boost)
        
        # Apply bounds
        estimated_gaussians = max(min_gaussians, min(estimated_gaussians, max_gaussians))
        
        # Enhanced reporting
        print(f"Enhanced image complexity analysis:")
        print(f"  - Visual complexity: {avg_complexity:.4f}")
        print(f"  - Depth complexity: {avg_depth:.4f}")
        print(f"  - Detail complexity: {avg_detail:.4f}")
        print(f"  - Motion complexity: {avg_motion:.4f}")
        print(f"  - High complexity ratio: {high_complexity_ratio:.2f}")
        print(f"  - Base gaussians: {base_estimate}")
        print(f"  - Complexity contribution: {int(complexity_contribution)}")
        print(f"  - Depth contribution: {int(depth_contribution)}")
        print(f"  - Detail contribution: {int(detail_contribution)}")
        print(f"  - Motion contribution: {int(motion_contribution)}")
        print(f"  - Outlier boost: {int(outlier_boost)}")
        print(f"  - Final estimate: {estimated_gaussians}")
        
        return estimated_gaussians
    
    def _analyze_image_complexity_enhanced(self, img_path):
        """
        Enhanced image complexity analysis with additional metrics.
        
        Returns:
            tuple: (visual_complexity, depth_complexity, detail_complexity, motion_complexity)
        """
        img = cv2.imread(img_path)
        if img is None:
            return 0, 0, 0, 0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Enhanced gradient analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Multiple edge thresholds for better analysis
        edge_density_low = np.sum(gradient_magnitude > 20) / (gray.shape[0] * gray.shape[1])
        edge_density_high = np.sum(gradient_magnitude > 50) / (gray.shape[0] * gray.shape[1])
        edge_complexity = (edge_density_low * 0.3 + edge_density_high * 0.7)
        
        # 2. Enhanced entropy calculation
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        non_zero_hist = hist[hist > 0]
        entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist)) / 8.0  # Normalized
        
        # 3. Enhanced frequency analysis
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        
        # Analyze different frequency bands
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # High frequency content (edges of spectrum)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[center_h-h//6:center_h+h//6, center_w-w//6:center_w+w//6] = 1
        low_freq = np.mean(magnitude_spectrum[mask == 1])
        high_freq = np.mean(magnitude_spectrum[mask == 0])
        freq_ratio = high_freq / (low_freq + 1e-6)
        
        # 4. Detail complexity using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        detail_complexity = np.std(laplacian) / 255.0
        
        # 5. Motion/blur estimation using variance of Laplacian
        motion_complexity = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0
        
        # 6. Color complexity
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_std = np.std(hsv, axis=(0, 1))
            color_complexity = np.mean(color_std) / 255.0
        else:
            color_complexity = 0.0
        
        # Combine metrics with enhanced weighting
        visual_complexity = (
            edge_complexity * 0.3 + 
            entropy * 0.25 + 
            freq_ratio * 0.2 + 
            color_complexity * 0.25
        )
        
        depth_complexity = np.std(gradient_magnitude) / 255.0
        
        return visual_complexity, depth_complexity, detail_complexity, motion_complexity
    
    # Keep all the existing helper methods from the original class
    def _read_ply_file_robust(self, ply_file_path):
        """
        Reads a PLY file (both ASCII and binary formats) and returns the point cloud data.
        """
        try:
            with open(ply_file_path, 'rb') as f:
                header_line = f.readline().decode('ascii', errors='ignore').strip()
                if header_line != 'ply':
                    raise ValueError(f"Not a valid PLY file: {ply_file_path}")
                
                is_binary = False
                vertex_count = 0
                has_colors = False
                header_lines = 1
                data_type = 'float'
                endianness = '<'
                property_names = []
                
                while True:
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    header_lines += 1
                    
                    if line == 'end_header':
                        break
                    
                    if line.startswith('format'):
                        if 'binary' in line:
                            is_binary = True
                            if 'binary_big_endian' in line:
                                endianness = '>'
                    
                    elif line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    
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
                
                points = np.zeros((vertex_count, 3), dtype=np.float32)
                colors = np.zeros((vertex_count, 3), dtype=np.uint8) if has_colors else None
                
                if is_binary:
                    x_idx = property_names.index('x') if 'x' in property_names else -1
                    y_idx = property_names.index('y') if 'y' in property_names else -1
                    z_idx = property_names.index('z') if 'z' in property_names else -1
                    r_idx = property_names.index('red') if 'red' in property_names else -1
                    g_idx = property_names.index('green') if 'green' in property_names else -1
                    b_idx = property_names.index('blue') if 'blue' in property_names else -1
                    
                    type_map = {
                        'float': 'f', 'float32': 'f', 'double': 'd', 'float64': 'd',
                        'uchar': 'B', 'uint8': 'B', 'char': 'b', 'int8': 'b',
                        'ushort': 'H', 'uint16': 'H', 'short': 'h', 'int16': 'h',
                        'uint': 'I', 'uint32': 'I', 'int': 'i', 'int32': 'i'
                    }
                    
                    property_formats = [type_map.get(data_type, 'f')] * len(property_names)
                    format_str = endianness + ''.join(property_formats)
                    property_size = struct.calcsize(format_str)
                    
                    try:
                        data_buffer = f.read(vertex_count * property_size)
                        
                        for i in range(vertex_count):
                            offset = i * property_size
                            
                            if offset + property_size > len(data_buffer):
                                print(f"Warning: Unexpected end of binary data at vertex {i}/{vertex_count}")
                                break
                                
                            values = struct.unpack_from(format_str, data_buffer, offset)
                            
                            if x_idx >= 0 and y_idx >= 0 and z_idx >= 0:
                                points[i, 0] = values[x_idx]
                                points[i, 1] = values[y_idx]
                                points[i, 2] = values[z_idx]
                            
                            if has_colors and r_idx >= 0 and g_idx >= 0 and b_idx >= 0:
                                colors[i, 0] = values[r_idx]
                                colors[i, 1] = values[g_idx]
                                colors[i, 2] = values[b_idx]
                    
                    except Exception as e:
                        print(f"Error parsing binary data: {e}")
                        print("Falling back to alternative approach...")
                        
                        if vertex_count > 0:
                            num_points = vertex_count
                            print(f"Using vertex count from header: {num_points}")
                            points = np.random.rand(num_points, 3) * 10 - 5
                            return points, colors
                else:
                    for i in range(vertex_count):
                        line = f.readline().decode('ascii', errors='ignore').strip()
                        values = line.split()
                        
                        points[i, 0] = float(values[0])
                        points[i, 1] = float(values[1])
                        points[i, 2] = float(values[2])
                        
                        if has_colors and len(values) >= 6:
                            colors[i, 0] = int(values[3])
                            colors[i, 1] = int(values[4])
                            colors[i, 2] = int(values[5])
                
                return points, colors
                
        except Exception as e:
            print(f"Error reading PLY file: {e}")
            
            try:
                file_size = os.path.getsize(ply_file_path)
                estimated_vertex_count = file_size // 50
                print(f"Attempting recovery: Estimated {estimated_vertex_count} vertices from file size")
                
                points = np.random.rand(estimated_vertex_count, 3) * 10 - 5
                return points, None
            except:
                pass
                
            return np.array([]), np.array([])
    
    def _read_colmap_points(self, points3D_file):
        """
        Read 3D points from a COLMAP points3D.txt file.
        """
        points = []
        colors = []
        
        try:
            with open(points3D_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                        
                        points.append([x, y, z])
                        colors.append([r, g, b])
            
            print(f"Successfully read {len(points)} points from {points3D_file}")
            return np.array(points), np.array(colors)
            
        except Exception as e:
            print(f"Error reading points3D file: {e}")
            return np.array([]), np.array([])
    
    def _analyze_point_quality(self, points3D_file):
        """
        Analyze the quality of the points based on reprojection errors.
        """
        errors = []
        
        try:
            with open(points3D_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        error = float(parts[7])
                        errors.append(error)
            
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Gaussian Estimator for Gaussian Splatting")
    
    # Main input arguments
    parser.add_argument("--colmap_dir", "-c", help="Path to COLMAP directory containing sparse/0 subfolder")
    parser.add_argument("--ply_file", "-p", help="Path to PLY point cloud file")
    parser.add_argument("--image_dir", "-i", help="Path to directory containing input images")
    
    # Enhanced method-specific parameters with new defaults
    parser.add_argument("--density_factor", "-d", type=float, default=8.0, 
                        help="Point density multiplier (enhanced default: 8.0)")
    parser.add_argument("--min_gaussians", "-min", type=int, default=500000, 
                        help="Minimum number of gaussians (enhanced default: 500k)")
    parser.add_argument("--max_gaussians", "-max", type=int, default=5000000, 
                        help="Maximum number of gaussians (enhanced default: 5M)")
    parser.add_argument("--sample_frames", "-s", type=int, default=15, 
                        help="Number of frames to sample for image analysis (enhanced default: 15)")
    
    # Enhanced options for image-based analysis
    parser.add_argument("--base_gaussians", "-b", type=int, default=500000, 
                        help="Base number of gaussians (enhanced default: 500k)")
    parser.add_argument("--complexity_factor", type=int, default=600000, 
                        help="Scaling factor for visual complexity (enhanced default: 600k)")
    parser.add_argument("--depth_factor", type=int, default=800000, 
                        help="Scaling factor for depth complexity (enhanced default: 800k)")
    
    # New advanced options
    parser.add_argument("--conservative_mode", action="store_true", 
                        help="Use original conservative estimation (reduces all estimates by 20%%)")
    parser.add_argument("--aggressive_mode", action="store_true", 
                        help="Use aggressive estimation (increases all estimates by 30%%)")
    parser.add_argument("--quality_target", choices=['fast', 'balanced', 'quality'], 
                        default='standard', help="Quality target affects estimation aggressiveness")
    
    args = parser.parse_args()
    
    # Validate that at least one input method is specified
    if not args.colmap_dir and not args.ply_file and not args.image_dir:
        print("Error: Must specify at least one of --colmap_dir, --ply_file, or --image_dir")
        parser.print_help()
        exit(1)
    
    estimator = GaussianEstimator()
    
    # Quality target adjustments
    quality_multipliers = {
        'draft': 0.7,
        'standard': 1.0,
        'high': 1.3,
    }
    quality_multiplier = quality_multipliers[args.quality_target]
    
    # Determine which method to use based on provided arguments
    if args.ply_file:
        print(f"Enhanced estimation from PLY file: {args.ply_file}")
        print(f"Quality target: {args.quality_target} (multiplier: {quality_multiplier:.1f})")
        estimated = estimator.estimate_from_ply(
            args.ply_file,
            density_factor=args.density_factor * quality_multiplier,
            min_gaussians=args.min_gaussians,
            max_gaussians=args.max_gaussians
        )
    elif args.colmap_dir:
        print(f"Enhanced estimation from COLMAP directory: {args.colmap_dir}")
        print(f"Quality target: {args.quality_target} (multiplier: {quality_multiplier:.1f})")
        estimated = estimator.estimate_from_colmap(
            args.colmap_dir,
            density_factor=args.density_factor * quality_multiplier,
            min_gaussians=args.min_gaussians,
            max_gaussians=args.max_gaussians
        )
    elif args.image_dir:
        print(f"Enhanced estimation from image directory: {args.image_dir}")
        print(f"Quality target: {args.quality_target} (multiplier: {quality_multiplier:.1f})")
        estimated = estimator.estimate_from_image_complexity(
            args.image_dir,
            sample_frames=args.sample_frames,
            base_gaussians=int(args.base_gaussians * quality_multiplier),
            complexity_factor=int(args.complexity_factor * quality_multiplier),
            depth_factor=int(args.depth_factor * quality_multiplier),
            min_gaussians=args.min_gaussians,
            max_gaussians=args.max_gaussians
        )
    
    # Apply mode-specific adjustments
    if args.conservative_mode:
        estimated = int(estimated * 0.8)
        print(f"Conservative mode applied: reduced to {estimated}")
    elif args.aggressive_mode:
        estimated = int(estimated * 1.3)
        print(f"Aggressive mode applied: increased to {estimated}")
    
    # Final bounds check
    estimated = max(args.min_gaussians, min(estimated, args.max_gaussians))
    
    print(f"\n{'='*60}")
    print(f"ENHANCED GAUSSIAN ESTIMATION RESULTS")
    print(f"{'='*60}")
    print(f"Final gaussian estimate: {estimated}")
    print(f"Recommended for MCMC Gaussian Splatting: --max-gaussians {estimated}")
    print(f"Recommended with 20%% buffer: --max-gaussians {int(estimated * 1.2)}")
    print(f"Recommended with 50%% buffer: --max-gaussians {int(estimated * 1.5)}")
    print(f"{'='*60}")
    
    # Additional recommendations based on estimate size
    if estimated < 800000:
        print("💡 Tip: This is a relatively small scene. Consider using higher quality settings.")
    elif estimated > 3000000:
        print("⚠️  Note: This is a complex scene. Ensure you have sufficient GPU memory.")
        print("   Consider using gradient clipping and lower learning rates.")
    
    print(f"\nQuality target used: {args.quality_target}")
    if args.conservative_mode:
        print("Mode: Conservative (20% reduction applied)")
    elif args.aggressive_mode:
        print("Mode: Aggressive (30% increase applied)")
    else:
        print("Mode: Enhanced (balanced estimation)")