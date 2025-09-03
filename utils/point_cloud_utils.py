from pathlib import Path
import sqlite3
import os, struct, numpy as np, math
import struct
import logging

logger = logging.getLogger(__name__)

class PointCloudUtils:

    @staticmethod
    def get_avg_features_from_colmap_db(db_path: str) -> tuple[int | None, float | None]:
        """
        Restituisce (avg_features, feature_consistency).
        - avg_features: numero medio di keypoints per immagine (int) oppure None.
        - feature_consistency: std/mean delle feature per immagine (float) oppure None.
        """
        try:
            p = Path(db_path)
            if not p.exists():
                logger.warning(f"⚠️ database.db non trovato: {db_path}")
                return None, None

            con = sqlite3.connect(str(p))
            cursor = con.cursor()
            cursor.execute("SELECT rows FROM keypoints WHERE data IS NOT NULL AND rows > 0")
            rows = cursor.fetchall()
            con.close()

            if not rows:
                return None, None

            feature_counts = [int(r[0]) for r in rows if r[0] and int(r[0]) > 0]
            if not feature_counts:
                return None, None

            avg_features = float(np.mean(feature_counts))
            std_features = float(np.std(feature_counts))
            feature_consistency = (std_features / avg_features) if avg_features > 0 else None

            return int(round(avg_features)), feature_consistency
        except Exception as e:
            logger.exception(f"❌ Errore lettura keypoints da DB: {e}")
            return None, None

    @staticmethod
    def count_points_from_points3D_bin(bin_path: str) -> int:
        """
        Ritorna il numero di punti leggendo l'header di points3D.bin (uint64 little-endian).
        """
        p = Path(bin_path)
        if not p.exists():
            raise FileNotFoundError(f"File non trovato: {bin_path}")
        with p.open("rb") as f:
            head = f.read(8)
            if len(head) != 8:
                raise ValueError(f"Header troppo corto in {bin_path}")
            (num_points,) = struct.unpack("<Q", head)
            return int(num_points)

    @staticmethod
    def compute_track_and_reproj_stats_from_bin(bin_path: str) -> dict:
        """
        COLMAP points3D.bin:
        [0:8)        num_points (uint64 LE)
        per punto:
            [0:8)      point3D_id (uint64)
            [8:32)     xyz (3 * double)
            [32:35)    rgb (3 * uint8)
            [35:43)    error (double)
            [43:51)    track_length (uint64)
            poi        track_length * 8 byte (image_id:uint32, point2D_idx:uint32)
        """
        track_lengths, reproj_errors = [], []

        with open(bin_path, "rb") as f:
            # 1) num_points
            b = f.read(8)
            if len(b) != 8:
                raise ValueError("Header troppo corto: manca num_points (8B)")
            (num_points,) = struct.unpack("<Q", b)

            for i in range(int(num_points)):
                # 2) header fisso del punto (51B)
                head = f.read(51)
                if len(head) < 51:
                    # file troncato
                    break

                # parse campi
                # id = struct.unpack("<Q", head[0:8])[0]  # se serve
                # xyz = struct.unpack("<3d", head[8:32]) # se serve
                # rgb = head[32:35]                      # se serve

                error = struct.unpack("<d", head[35:43])[0]
                track_length = struct.unpack("<Q", head[43:51])[0]

                # sanity check (limiti ragionevoli)
                if not (math.isfinite(error) and -1e6 < error < 1e6):
                    # se vuoi, logga e interrompi/parsa conservativamente
                    break
                if track_length > 1_000_000:
                    # quasi certamente file corrotto o misallineato
                    break

                reproj_errors.append(float(error))
                track_lengths.append(int(track_length))

                # 3) salta il blocco track: 8B per osservazione (2 x uint32)
                skip = int(track_length) * 8
                f.seek(skip, os.SEEK_CUR)

        if not track_lengths:
            return {
                "avg_track_length": None,
                "median_track_length": None,
                "median_reproj_error_px": None,
                "num_points": 0
            }

        return {
            "avg_track_length": float(np.mean(track_lengths)),
            "median_track_length": float(np.median(track_lengths)),
            "median_reproj_error_px": float(np.median(reproj_errors)) if reproj_errors else None,
            "num_points": len(track_lengths),
        }


    @staticmethod
    def count_points_from_ply(ply_path: str) -> int:
        """
        Legge l'header del PLY e ritorna il numero di vertici (element vertex N).
        """
        p = Path(ply_path)
        if not p.exists():
            raise FileNotFoundError(f"PLY non trovato: {ply_path}")

        n_vertices = None
        with p.open("rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                s = line.decode("utf-8", errors="ignore").strip()
                if s.startswith("element vertex"):
                    parts = s.split()
                    try:
                        n_vertices = int(parts[-1])
                    except Exception:
                        pass
                if s == "end_header":
                    break

        if n_vertices is None:
            raise ValueError("Impossibile determinare il numero di punti dal PLY (manca 'element vertex N').")
        return n_vertices

    @staticmethod
    def compute_track_and_reproj_stats(txt_path: str) -> dict:
        """
        Calcola statistiche dal points3D.txt:
          - avg_track_length
          - median_track_length
          - median_reproj_error_px
        """
        track_lengths, reproj_errors = [], []
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 9:
                        try:
                            reproj_errors.append(float(parts[8]))
                        except Exception:
                            pass
                    if len(parts) > 8:
                        tlen = (len(parts) - 8) // 2
                        track_lengths.append(tlen)
        except Exception as e:
            logger.exception(f"❌ Errore lettura track/reproj da {txt_path}: {e}")
            return {"avg_track_length": None, "median_track_length": None, "median_reproj_error_px": None}

        avg_t = float(np.mean(track_lengths)) if track_lengths else None
        med_t = float(np.median(track_lengths)) if track_lengths else None
        med_err = float(np.median(reproj_errors)) if reproj_errors else None
        return {
            "avg_track_length": avg_t,
            "median_track_length": med_t,
            "median_reproj_error_px": med_err
        }

    def aggregate_reconstruction_stats(self, model_dir: str) -> dict:
        """
        Aggrega statistiche dalla ricostruzione COLMAP:
          - generated_points
          - avg_features, feature_consistency
          - avg_track_length, median_track_length
          - median_reproj_error_px
        """
        model_dir_p = Path(model_dir)

        # DB
        db_candidates = [
            model_dir_p / "distorted" / "database.db",
            model_dir_p / "database.db",
        ]
        db_path = next((p for p in db_candidates if p.exists()), None)
        avg_features, feature_consistency = (None, None)
        if db_path:
            avg_features, feature_consistency = self.get_avg_features_from_colmap_db(str(db_path))

        # points3D
        bin_candidates = [
            model_dir_p / "sparse" / "0" / "points3D.bin",
            model_dir_p / "sparse" / "points3D.bin",
        ]
        ply_candidates = [
            model_dir_p / "sparse" / "0" / "points3D.ply",
            model_dir_p / "sparse" / "points3D.ply",
        ]
        txt_candidates = [
            model_dir_p / "sparse" / "0" / "points3D.txt",
            model_dir_p / "sparse" / "points3D.txt",
        ]

        generated_points = None
        bin_path = next((p for p in bin_candidates if p.exists()), None)
        if bin_path:
            try:
                generated_points = self.count_points_from_points3D_bin(str(bin_path))
            except Exception as e:
                logger.exception(f"❌ Errore lettura {bin_path}: {e}")
        else:
            ply_path = next((p for p in ply_candidates if p.exists()), None)
            if ply_path:
                try:
                    generated_points = self.count_points_from_ply(str(ply_path))
                except Exception as e:
                    logger.exception(f"❌ Errore lettura {ply_path}: {e}")

        if generated_points is None:
            generated_points = 0

        # track/reproj
        track_stats = {"avg_track_length": None, "median_track_length": None, "median_reproj_error_px": None}
        txt_path = next((p for p in txt_candidates if p.exists()), None)
        bin_path = next((p for p in bin_candidates if p.exists()), None)
        if txt_path:
            track_stats = self.compute_track_and_reproj_stats(str(txt_path))
        elif bin_path:
            track_stats = self.compute_track_and_reproj_stats_from_bin(str(bin_path))

        summary = {
            "generated_points": generated_points,
            "avg_features": avg_features,
            "feature_consistency": feature_consistency,
            **track_stats,
        }
        return summary
