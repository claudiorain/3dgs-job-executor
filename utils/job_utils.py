import re
import os



class JobUtils:

    def _extract_metrics_from_section(self, metrics_section):
        """Estrai metriche da una sezione specifica"""
        return {
            "ssim": metrics_section.get("SSIM", None),
            "psnr": metrics_section.get("PSNR", None), 
            "lpips": metrics_section.get("LPIPS", None)
        }
    
    def find_latest_iteration_folder(self,output_dir):
        """Trova la cartella iteration_XXXXX con il numero più alto"""
        point_cloud_dir = os.path.join(output_dir, "point_cloud")
        
        if not os.path.exists(point_cloud_dir):
            return None
        
        # Pattern per cartelle iteration_NUMERO
        iteration_pattern = re.compile(r'^iteration_(\d+)$')
        max_iteration = -1
        latest_folder = None
        
        for folder_name in os.listdir(point_cloud_dir):
            match = iteration_pattern.match(folder_name)
            if match:
                iteration_num = int(match.group(1))
                if iteration_num > max_iteration:
                    max_iteration = iteration_num
                    latest_folder = folder_name
        
        if latest_folder:
            ply_path = os.path.join(point_cloud_dir, latest_folder, "point_cloud.ply")
            print(f"🎯 Latest iteration folder: {latest_folder}")
            print(f"📄 Expected PLY path: {ply_path}")
            
            if os.path.exists(ply_path):
                file_size = os.path.getsize(ply_path)
                print(f"✅ PLY file found: {file_size} bytes")
                return ply_path
            else:
                print(f"❌ PLY file doesn't exist at: {ply_path}")
                return None
        else:
            print(f"❌ No iteration folders found")
            return None

    def get_colmap_reconstruction_stats(model_dir):
        """
        Raccoglie il numero di punti 3D dalla ricostruzione COLMAP
        """
        import struct
        
        sparse_path = os.path.join(model_dir, "sparse", "0")
        points_3d = 0
        
        # Conta punti 3D
        points_bin = os.path.join(sparse_path, "points3D.bin")
        points_txt = os.path.join(sparse_path, "points3D.txt")
        
        if os.path.exists(points_bin):
            try:
                with open(points_bin, 'rb') as f:
                    points_3d = struct.unpack('<Q', f.read(8))[0]
            except:
                pass
        elif os.path.exists(points_txt):
            try:
                with open(points_txt, 'r') as f:
                    count = 0
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            count += 1
                    points_3d = count
            except:
                pass
        
        return points_3d