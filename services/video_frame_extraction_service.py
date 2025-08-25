"""
Video Frame Extraction Service — selezione ADATTIVA: prova best-n,
valuta; se serve prova batched e sceglie il migliore. Gestisce cartelle
temporanee e pulizia della cartella finale.
"""
import os, cv2, subprocess, shutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# =======================
# PARAMS
# =======================
@dataclass
class FrameExtractionParams:
    target_width: int
    target_height: int
    target_frame_count: int = 200
    # Parametri interni (non esposti al chiamante):
    # best-n
    min_buffer: str = "1"
    # batched (predefiniti "moderati")
    batch_size_default: int = 5
    batch_buffer_default: int = 2
    # analisi rapida
    quick_pairs: int = 60     # quante coppie analizzare (uniform sampling)
    quick_resize: int = 640   # ridimensionamento max lato lungo per analisi

@dataclass
class FrameExtractionResult:
    success: bool
    extracted_frame_count: int
    frame_files: list
    extraction_params: Dict[str, Any]
    error_message: Optional[str] = None
    processing_stats: Optional[Dict[str, Any]] = None

class VideoFrameExtractionService:
    def __init__(self): pass

    # --------------------- Public entry ---------------------
    def extract_frames(self, video_path: str, output_directory: str,
                       extraction_params: FrameExtractionParams) -> FrameExtractionResult:
        try:
            print(f"🎬 Starting adaptive frame extraction for: {video_path}")

            if not self._validate_inputs(video_path, output_directory):
                return FrameExtractionResult(False, 0, [], {}, "Input validation failed")

            video_analysis = self._analyze_video(video_path)
            print(f"📊 Video analysis: {video_analysis}")

            # Cartelle temporanee per i due metodi
            out_final = Path(output_directory)
            out_bestn = out_final.parent / (out_final.name + "_tmp_bestn")
            out_batched = out_final.parent / (out_final.name + "_tmp_batched")

            # Assicurati che le tmp siano pulite
            self._ensure_empty_dir(out_bestn)
            self._ensure_empty_dir(out_batched)

            # 1) BEST-N (baseline)
            opt_bestn = self._calc_params(video_analysis, extraction_params, method="best-n")
            print(f"🎯 Params (best-n): {opt_bestn}")
            res_bestn = self._run_sharp_frames(video_path, out_bestn, opt_bestn)
            if not res_bestn['success']:
                raise RuntimeError(f"Sharp-frames (best-n) failed: {res_bestn.get('error_message','')}")

            frames_bestn = self._list_frames(out_bestn)
            metrics_bestn = self._quick_analyze(frames_bestn, extraction_params)
            print(f"📈 Quick metrics (best-n): {metrics_bestn}")

            # Decisione: se il baseline è “sano”, accettiamo; altrimenti proviamo batched
            need_batched = self._needs_batched(metrics_bestn)

            metrics_batched = None
            frames_batched = []
            opt_batched = None
            if need_batched:
                opt_batched = self._calc_params(video_analysis, extraction_params, method="batched")
                print(f"🎯 Params (batched): {opt_batched}")
                res_batched = self._run_sharp_frames(video_path, out_batched, opt_batched)
                if not res_batched['success']:
                    print("⚠️ Batched failed, falling back to best-n.")
                else:
                    frames_batched = self._list_frames(out_batched)
                    metrics_batched = self._quick_analyze(frames_batched, extraction_params)
                    print(f"📈 Quick metrics (batched): {metrics_batched}")

            # Scelta finale
            choice = self._choose_best(metrics_bestn, metrics_batched)
            print(f"✅ Selected method: {choice}")

            # Copia nella cartella finale (svuotandola)
            self._ensure_empty_dir(out_final)
            chosen_dir = out_bestn if choice == "best-n" else out_batched
            self._copy_dir_images(chosen_dir, out_final)

            # Pulizia tmp
            self._safe_rmtree(out_bestn)
            self._safe_rmtree(out_batched)

            # Risultato finale
            final_frames = self._list_frames(out_final)
            processing_stats = {
                'video_analysis': video_analysis,
                'chosen_method': choice,
                'bestn_params': opt_bestn,
                'bestn_metrics': metrics_bestn,
                'batched_params': opt_batched,
                'batched_metrics': metrics_batched
            }
            print(f"✅ Frame extraction completed successfully: {len(final_frames)} frames")

            return FrameExtractionResult(
                True, len(final_frames), [str(p) for p in final_frames],
                extraction_params={ 'chosen_method': choice, **(opt_batched if choice=="batched" else opt_bestn) },
                processing_stats=processing_stats
            )

        except Exception as e:
            print(f"❌ Adaptive frame extraction failed: {str(e)}")
            return FrameExtractionResult(False, 0, [], {}, f"Extraction failed: {str(e)}")

    # --------------------- Helpers: IO & validation ---------------------
    def _validate_inputs(self, video_path: str, output_directory: str) -> bool:
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}"); return False
        parent = os.path.dirname(output_directory) or "."
        if not os.path.exists(parent):
            try: os.makedirs(parent, exist_ok=True)
            except Exception as e:
                print(f"❌ Cannot create parent dir '{parent}': {e}"); return False
        if not os.access(parent, os.W_OK):
            print(f"❌ Output directory not writable: {output_directory}"); return False
        cap = cv2.VideoCapture(video_path); ok = cap.isOpened(); cap.release()
        if not ok: print(f"❌ Cannot open video file: {video_path}")
        return ok

    def _ensure_empty_dir(self, p: Path):
        if p.exists(): self._safe_rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    def _safe_rmtree(self, p: Path):
        try:
            if p.exists(): shutil.rmtree(p)
        except Exception as e:
            print(f"⚠️ Could not remove {p}: {e}")

    def _copy_dir_images(self, src: Path, dst: Path):
        for f in sorted(src.iterdir()):
            if f.suffix.lower() in (".jpg",".jpeg",".png"):
                shutil.copy2(str(f), str(dst / f.name))

    def _list_frames(self, p: Path) -> List[Path]:
        return sorted([q for q in p.iterdir() if q.suffix.lower() in (".jpg",".jpeg",".png")], key=lambda x: x.name)

    # --------------------- Video analysis ---------------------
    def _analyze_video(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (n / fps) if fps > 0 else 0.0
        cap.release()
        is_portrait = h > w
        print(f"📺 Original resolution: {w}x{h}")
        print(f"📐 Orientation: {'Portrait' if is_portrait else 'Landscape'}")
        print(f"🎞️ Duration: {duration:.2f}s | 🎥 Original FPS: {fps:.2f}")
        return {
            'original_width': w, 'original_height': h,
            'video_fps': fps, 'total_frames': n, 'duration_seconds': duration,
            'is_portrait': is_portrait, 'orientation': 'Portrait' if is_portrait else 'Landscape',
            'aspect_ratio': (w / h) if h > 0 else 1.0
        }

    # --------------------- Parametri per i metodi ---------------------
    def _calc_params(self, va: Dict[str, Any], p: FrameExtractionParams, method: str) -> Dict[str, Any]:
        final_width = p.target_height if va['is_portrait'] else p.target_width
        duration, video_fps = va['duration_seconds'], va['video_fps']
        target = p.target_frame_count

        if duration > 0:
            if method == "batched":
                step = max(1, int(p.batch_size_default) + int(p.batch_buffer_default))
                want_fps = (target * step) / duration
            else:
                want_fps = target / duration
            # clamp
            clamped = min(want_fps, video_fps if video_fps > 0 else want_fps)
            if video_fps > 0 and want_fps > video_fps:
                print(f"⚠️ Requested FPS {want_fps:.2f} clamped to source FPS {video_fps:.2f}")
            want_fps = max(0.5, clamped)
        else:
            want_fps = 1.0

        return {
            'method': method,
            'final_width': final_width,
            'extraction_fps': int(round(want_fps)),
            'min_buffer': p.min_buffer,
            'batch_size': p.batch_size_default,
            'batch_buffer': p.batch_buffer_default,
            'target_frame_count': target,
            'calculated_extraction_fps_precise': want_fps,
            'width_adaptation_applied': va['is_portrait']
        }

    # --------------------- Invocazione sharp-frames ---------------------
    def _run_sharp_frames(self, video_path: str, out_dir: Path, opt: Dict[str, Any]) -> Dict[str, Any]:
        cmd = [
            "sharp-frames", video_path, str(out_dir),
            "--selection-method", opt['method'],
            "--fps", str(opt['extraction_fps']),
            "--width", str(opt['final_width']),
            "--format", "jpg", "--force-overwrite"
        ]
        if opt['method'] == "best-n":
            cmd += ["--min-buffer", str(opt['min_buffer'])]
        elif opt['method'] == "batched":
            cmd += ["--batch-size", str(opt['batch_size']), "--batch-buffer", str(opt['batch_buffer'])]

        print(f"🔧 Executing Sharp-Frames: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Sharp-frames completed successfully")
            print(f"📝 stdout: {result.stdout}")
            if result.stderr: print(f"⚠️ Sharp-frames warnings: {result.stderr}")
            return {'success': True, 'stdout': result.stdout, 'stderr': result.stderr, 'return_code': result.returncode}
        except subprocess.CalledProcessError as e:
            print(f"❌ Sharp-frames failed with exit code {e.returncode}")
            print(f"📝 stdout: {e.stdout}")
            print(f"🔴 stderr: {e.stderr}")
            return {'success': False, 'stdout': e.stdout, 'stderr': e.stderr,
                    'return_code': e.returncode, 'error_message': f"Sharp-frames error: {e.stderr}"}

    # --------------------- Analisi rapida (SSIM + overlap ORB) ---------------------
    def _quick_analyze(self, frames: List[Path], p: FrameExtractionParams) -> Dict[str, Any]:
        if len(frames) < 2:
            return {'n_frames': len(frames), 'n_pairs': 0, 'ssim_median': None,
                    'ssim_p95': None, 'overlap_median': None}

        # campiona coppie uniformemente
        pairs = list(zip(frames[:-1], frames[1:]))
        if len(pairs) > p.quick_pairs:
            idx = np.linspace(0, len(pairs)-1, p.quick_pairs, dtype=int)
            pairs = [pairs[i] for i in idx]

        def _read_gray_scaled(fp: Path):
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None: raise IOError(f"Cannot read {fp}")
            h, w = img.shape[:2]
            scale = p.quick_resize / max(h, w) if max(h, w) > p.quick_resize else 1.0
            if scale < 1.0:
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            return img

        # ORB per velocità
        det = cv2.ORB_create(nfeatures=1000)
        ssim_vals, overlaps = [], []

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        for f1, f2 in pairs:
            a = _read_gray_scaled(f1)
            b = _read_gray_scaled(f2)
            # SSIM
            ssim_val = self._ssim(a, b)
            ssim_vals.append(ssim_val)
            # overlap con ratio-test
            kp1, des1 = det.detectAndCompute(a, None)
            kp2, des2 = det.detectAndCompute(b, None)
            good = 0
            if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:
                matches = bf.knnMatch(des1, des2, k=2)
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good += 1
            denom = max(min(len(kp1 or []), len(kp2 or [])), 1)
            overlaps.append(good / denom)

        arr = np.array(ssim_vals, dtype=float)
        ov  = np.array(overlaps, dtype=float)

        return {
            'n_frames': len(frames),
            'n_pairs': len(pairs),
            'ssim_median': float(np.median(arr)) if arr.size else None,
            'ssim_p95': float(np.quantile(arr, 0.95)) if arr.size else None,
            'overlap_median': float(np.median(ov)) if ov.size else None
        }

    def _ssim(self, a: np.ndarray, b: np.ndarray) -> float:
        # Ridimensiona a dimensioni comuni
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
            a = cv2.resize(a, (w, h)); b = cv2.resize(b, (w, h))
        # SSIM "light" (stima) con OpenCV: usiamo MSE->PSNR proxy se vuoi
        # Ma preferiamo una stima diretta: normalizziamo e usiamo correlazione come proxy
        a_f = a.astype(np.float32); b_f = b.astype(np.float32)
        a_f = (a_f - a_f.mean()) / (a_f.std() + 1e-6)
        b_f = (b_f - b_f.mean()) / (b_f.std() + 1e-6)
        corr = float((a_f * b_f).mean())
        # Mappa correlazione [-1,1] ~> [0,1] come proxy SSIM (veloce)
        return (corr + 1.0) * 0.5

    # --------------------- Policy di decisione ---------------------
    def _needs_batched(self, m: Dict[str, Any]) -> bool:
        """Regole semplici:
           - troppo 'saltellante' se overlap<0.30 o ssim_median<0.20
           - troppo 'ridondante' se ssim_p95>0.90 (qui meglio ridurre densità, ma batched 3/1 aiuta)
        """
        if m['n_pairs'] == 0: return False
        if (m['overlap_median'] is not None and m['overlap_median'] < 0.30): return True
        if (m['ssim_median']  is not None and m['ssim_median']  < 0.20): return True
        if (m['ssim_p95']     is not None and m['ssim_p95']     > 0.90): return True
        return False

    def _choose_best(self, m_bestn: Dict[str, Any], m_batched: Optional[Dict[str, Any]]) -> str:
        """Scoring semplice: preferisci la soluzione con overlap_median più alto
           mantenendo ssim_p95 < 0.90; a parità, tieni best-n.
        """
        if m_batched is None: return "best-n"
        # penalizza ridondanza eccessiva
        def score(m):
            if m is None: return -1e9
            if m['overlap_median'] is None: return -1e9
            pen = 0.0
            if m['ssim_p95'] is not None and m['ssim_p95'] > 0.90:
                pen -= 0.5
            base = m['overlap_median']
            return base + pen

        s_bestn  = score(m_bestn)
        s_batched = score(m_batched)
        return "batched" if s_batched > s_bestn else "best-n"
