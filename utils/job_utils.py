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

