import re
import os
from pathlib import Path
import sqlite3
import numpy as np


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

    def get_colmap_reconstruction_stats(self,model_dir):
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
    
    def get_avg_features_from_colmap_db(db_path: str) -> float | None:
        """
        Restituisce il numero medio di keypoints per immagine dalla tabella 'keypoints'
        del database COLMAP (database.db). Ritorna None se la query fallisce.
        """
        try:
            con = sqlite3.connect(db_path)
            cursor = con.cursor()
            
            # Analizza keypoints usando la struttura documentata COLMAP
            # La tabella keypoints ha: image_id, rows, cols, data
            cursor.execute("SELECT image_id, rows FROM keypoints WHERE data IS NOT NULL AND rows > 0")
            keypoint_data = cursor.fetchall()

            # Calcola statistiche delle feature (rows = numero di keypoints)
            feature_counts = [int(row[1]) for row in keypoint_data if row[1] > 0]

            avg_features = np.mean(feature_counts)
            std_features = np.std(feature_counts)
            feature_consistency = std_features / avg_features if avg_features > 0 else 1

            con.close()
            return int(avg_features) if avg_features is not None else None
        except Exception:
            return None
    
    def count_points_from_ply(ply_path: str) -> int:
        """
        Legge l'header del PLY e ritorna il numero di vertici (element vertex N).
        Funziona sia per PLY ASCII che binari, perché il conteggio è nell'header.
        """
        p = Path(ply_path)
        if not p.exists():
            raise FileNotFoundError(f"PLY non trovato: {ply_path}")

        n_vertices = None
        with p.open("rb") as f:
            # Leggi header riga per riga fino a 'end_header'
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    s = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    s = ""
                if s.startswith("element vertex"):
                    parts = s.split()
                    # atteso: ["element", "vertex", "<N>"]
                    if len(parts) >= 3 and parts[2].isdigit():
                        n_vertices = int(parts[2])
                    else:
                        # prova pars. più robusta
                        try:
                            n_vertices = int(parts[-1])
                        except Exception:
                            pass
                if s == "end_header":
                    break

        if n_vertices is None:
            raise ValueError("Impossibile determinare il numero di punti dal PLY (header senza 'element vertex N').")

        return n_vertices
    
    def estimate_final_gaussians(
    generated_points: int,
    avg_features: float | None = None,
    base_ratio: float = 16.5) -> int:
        """
        Versione con smoothing morbido su avg_features.
        - base_ratio: 16.5 (come fallback/ancora)
        - Se avg_features è disponibile, applica un fattore di complessità in [0.9, 1.2]
        mappando avg_features da [4000, 10000] -> [1.2, 0.9]
        (scene povere di feature => +20%, ricche => -10%)
        Ritorna: (stima_gaussiane, complexity_factor_usato)
        """
        if avg_features is None:
            return int(generated_points * base_ratio)

        lo, hi = 4000.0, 10000.0
        x = max(lo, min(hi, float(avg_features)))
        t = (x - lo) / (hi - lo)            # 0..1
        complexity_factor = 1.2 * (1 - t) + 0.9 * t  # 1.2 -> 0.9

        estimate = int(generated_points * base_ratio * complexity_factor)
        return estimate
        
    def list_image_files(input_dir: str, valid_ext):
        """Ritorna solo i path delle immagini valide nella cartella indicata (non scende in /images)."""
        p = Path(input_dir)
        files = [
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in valid_ext
        ]
        return files

    def safe_get(d: dict, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur