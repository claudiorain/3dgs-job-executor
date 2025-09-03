from config.message_queue import get_connection,get_channel  # Assicurati che questa funzione restituisca il client del database
from services.model_service import ModelService
from services.repository_service import RepositoryService
from services.training_params_service import TrainingParamsService,QualityLevel
from models.model import Engine
from converters.ply_to_ksplat_converter import process_ply_to_ksplat,save_splat_file
from datetime import datetime
import os
import json
import requests
import shutil
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import tempfile
from utils.phase_zip_helper import PhaseZipHelper
from utils.job_utils import JobUtils
from utils.point_cloud_utils import PointCloudUtils
from utils.vram_monitor import VRAMMonitor  
from plyfile import PlyData
from services.gaussian_estimator_service import estimate_gaussians_from_stats
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


model_service = ModelService()
repository_service = RepositoryService()
training_params_service = TrainingParamsService()
phase_zip_helper = PhaseZipHelper()
job_utils = JobUtils()

point_cloud_utils = PointCloudUtils()


# Cartella per i file di staging (zip delle fasi)
S3_STAGING_PREFIX = os.getenv('S3_STAGING_PREFIX', 'staging')

# Cartella per i deliverable finali
S3_DELIVERY_PREFIX = os.getenv('S3_DELIVERY_PREFIX', 'delivery')

WORKING_DIR = os.getenv("MODEL_WORKING_DIR") 
COLMAP_API_URL = "http://colmap-converter-api:8060"

POINT_CLOUD_BUILDING_PHASE_ZIP_NAME = "point_cloud_building_phase.zip"
TRAINING_PHASE_ZIP_NAME = "training_phase.zip"

# Creiamo la mappa che associa ciascun valore a un oggetto con api-url
engine_map = {
    'INRIA': { 'api-url': 'http://gaussian-splatting-api:8100' },
    'MCMC': { 'api-url': 'http://3dgs-mcmc-api:8101' },
    'TAMING': { 'api-url': 'http://taming-3dgs-api:8102' },
}

# Assicurati che la cartella esista
os.makedirs(WORKING_DIR, exist_ok=True)

class QueueJobService:

    def __init__(self):
        """Inizializza la connessione a RabbitMQ"""
        self.connection = get_connection()
        self.channel = get_channel(self.connection)

    def fail(self,model_id: str,phase_str : str,error_message: str):
        print(error_message)
        model_service.fail_phase(model_id, phase_str,error_message)
        model_service.update_model_status(model_id, {"overall_status": "FAILED"})
        
    async def handle_frame_extraction(self, ch, method, model_id, data):
        """
        Fase 1: se video_s3_key è un video -> estrazione frame.
                se video_s3_key è uno zip   -> unzip diretto dei frame (bypassa estrazione).
        """
        import os
        import zipfile
        from glob import glob
        from datetime import datetime

        from services.video_frame_extraction_service import (
            VideoFrameExtractionService,
            FrameExtractionParams,
        )

        IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        def _list_frame_files(base_dir: str):
            """Raccoglie immagini (case-insensitive) in modo ricorsivo e le ordina per nome."""
            exts = {e.lower() for e in IMAGE_EXTS}
            files = []
            for root, _dirs, filenames in os.walk(base_dir):
                for fname in filenames:
                    if os.path.splitext(fname)[1].lower() in exts:
                        files.append(os.path.join(root, fname))
            return sorted(files, key=lambda p: p.lower())

        model_service.start_phase(model_id, "frame_extraction")
        model = model_service.get_model_by_id(model_id)

        if not model:
            self.fail(model_id, "frame_extraction", f"Error: No model found for model_id {model_id}")
            return False

        video_s3_key = model.video_s3_key
        if not video_s3_key:
            self.fail(model_id, "frame_extraction", f"Error: No video_s3_key found for model_id {model_id}")
            return False

        model_dir = os.path.join(WORKING_DIR, f"{model_id}")
        os.makedirs(model_dir, exist_ok=True)

        local_video_path = os.path.join(model_dir, "input_video.mp4")   # usato solo se è un video
        local_zip_path   = os.path.join(model_dir, "input_frames.zip")  # usato solo se è uno zip
        frames_output_folder = os.path.join(model_dir, "input")
        os.makedirs(frames_output_folder, exist_ok=True)

        # Variabili comuni per i metadati finali
        frame_files = []
        extracted_frame_count = 0
        extraction_parameters = None
        processing_stats = {}
        frame_extraction_seconds = 0.0

        try:
            is_zip = str(video_s3_key).lower().endswith(".zip")

            if is_zip:
                # ========= FLUSSO ZIP: niente estrazione, solo unzip =========
                print(f"📥 Downloading frames ZIP from S3: {video_s3_key}")
                repository_service.download(video_s3_key, local_zip_path)

                if not os.path.exists(local_zip_path):
                    raise Exception(f"Frames ZIP not found after download: {local_zip_path}")

                archive_size = os.path.getsize(local_zip_path)
                print(f"📦 Frames ZIP downloaded: {archive_size / 1024 / 1024:.2f} MB")

                unzip_start = datetime.utcnow()
                print(f"🗜️  Unzipping frames into: {frames_output_folder} ...")
                with zipfile.ZipFile(local_zip_path, "r") as zf:
                    zf.extractall(frames_output_folder)
                unzip_end = datetime.utcnow()

                frame_extraction_seconds = (unzip_end - unzip_start).total_seconds()

                # Raccogli i frame scompattati
                frame_files = _list_frame_files(frames_output_folder)
                extracted_frame_count = len(frame_files)

                if extracted_frame_count == 0:
                    raise Exception("No frames found after unzip.")

                print(f"✅ Frames ready (from ZIP): {extracted_frame_count}")

                # Parametri/metrica coerenti con il ramo video, ma marcati come 'pre-extracted'
                extraction_parameters = {
                    "source": "pre-extracted-zip",
                    "target_width": None,
                    "target_height": None,
                    "target_frames": extracted_frame_count,
                    "selection_method": None,
                }
                processing_stats = {
                    "unzipped_bytes": archive_size,
                    "unzipped_seconds": frame_extraction_seconds,
                }

            else:
                # ========= FLUSSO VIDEO: estrazione frame =========
                print(f"📥 Downloading video from S3: {video_s3_key}")
                repository_service.download(video_s3_key, local_video_path)

                if not os.path.exists(local_video_path):
                    raise Exception(f"Video file not found after download: {local_video_path}")

                video_size = os.path.getsize(local_video_path)
                print(f"📊 Video downloaded successfully: {video_size / 1024 / 1024:.2f} MB")

                print(f"🔄 Avvio frame extraction per model_id {model_id}...")
                frame_extraction_start_time = datetime.utcnow()

                # Recupero configurazione solo per il ramo video
                engine = model.training_config.get("engine") if model.training_config else None
                quality_level = model.training_config.get("quality_level") if model.training_config else None

                if not engine:
                    self.fail(model_id, "frame_extraction", f"Error: No engine found in model {model_id}")
                    return False

                generated_params = training_params_service.generate_params(Engine(engine), QualityLevel(quality_level))
                print(f"🎯 Generated training parameters completed")

                target_width = generated_params.preprocessing_params.get("target_width", 1280)
                target_height = generated_params.preprocessing_params.get("target_height", 720)
                target_frame_count = generated_params.preprocessing_params.get("target_frames", 200)

                print(f"🔍 Frame extraction config - Width: {target_width}, Height: {target_height}, Frames: {target_frame_count}")

                frame_extraction_service = VideoFrameExtractionService()
                extraction_params = FrameExtractionParams(
                    target_width=target_width,
                    target_height=target_height,
                    target_frame_count=target_frame_count,
                    # niente selection_method qui
                )
                result = frame_extraction_service.extract_frames(
                    video_path=local_video_path,
                    output_directory=frames_output_folder,
                    extraction_params=extraction_params,
                )

                if not result.success:
                    raise Exception(f"Frame extraction failed: {result.error_message}")

                frame_extraction_end_time = datetime.utcnow()
                frame_extraction_seconds = (frame_extraction_end_time - frame_extraction_start_time).total_seconds()

                frame_files = result.frame_files or _list_frame_files(frames_output_folder)
                extracted_frame_count = result.extracted_frame_count or len(frame_files)

                extraction_parameters = result.extraction_params
                processing_stats = result.processing_stats

                print(f"✅ Atomic frame extraction completed: {extracted_frame_count} frames")

            # ===== THUMBNAIL =====
            thumbnail_suffix = "thumbnail.jpg"
            thumbnail_s3_key = f"{S3_DELIVERY_PREFIX}/{model_id}/{thumbnail_suffix}"

            if frame_files:
                first_frame = frame_files[0]
                print(f"📸 Using first frame as thumbnail: {first_frame}")
                if os.path.exists(first_frame):
                    try:
                        repository_service.upload(first_frame, thumbnail_s3_key)
                        print(f"✅ Thumbnail uploaded to S3: {thumbnail_s3_key}")
                    except Exception as e:
                        print(f"❌ Error uploading thumbnail: {e}")
                        thumbnail_suffix = None
                else:
                    print(f"⚠️ Thumbnail file not found: {first_frame}")
                    thumbnail_suffix = None
            else:
                print("⚠️ No frames available for thumbnail")
                thumbnail_suffix = None

            # ===== UPLOAD ZIP DI FASE SU S3 =====
            print("📤 Uploading frame extraction results to S3...")
            is_zip_uploaded = phase_zip_helper.create_phase_zip_and_upload(
                model_id, model_dir, POINT_CLOUD_BUILDING_PHASE_ZIP_NAME, ["input"]
            )
            if not is_zip_uploaded:
                self.fail(model_id, "frame_extraction", f"Error: Failed to create phase ZIP for model_id {model_id}")
                return False

            # ===== METADATI FASE =====
            # per robustezza evitiamo divisioni per zero
            frame_per_second = frame_extraction_seconds / max(extracted_frame_count, 1)

            phase_metadata = {
                "frame_count": extracted_frame_count,
                # manteniamo il nome del campo, anche se in ramo zip contiene il peso dell'archivio
                "video_size_mb": (os.path.getsize(local_video_path) / 1024 / 1024) if os.path.exists(local_video_path) else (
                    os.path.getsize(local_zip_path) / 1024 / 1024 if os.path.exists(local_zip_path) else None
                ),
                "extraction_parameters": extraction_parameters,
                "processing_stats": processing_stats,
                "frame_extraction_seconds": frame_extraction_seconds,
                "frame_per_second": frame_per_second,
                "source_type": "zip" if is_zip else "video",
            }

            if thumbnail_suffix:
                model_service.update_model_status(model_id, {"thumbnail_suffix": thumbnail_suffix})

            model_service.complete_phase(model_id, "frame_extraction", "PENDING", metadata=phase_metadata)

            print("✅ Frame extraction phase completed successfully")
            return True

        except Exception as e:
            print(f"❌ Error in frame extraction handler: {e}")
            self.fail(model_id, "frame_extraction", f"Frame extraction failed: {e}")
            return False

        finally:
            # ===== CLEANUP =====
            try:
                if os.path.exists(local_video_path):
                    os.remove(local_video_path)
                    print(f"🧹 Cleaned up local video file: {local_video_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not cleanup video file: {e}")
            try:
                if os.path.exists(local_zip_path):
                    os.remove(local_zip_path)
                    print(f"🧹 Cleaned up local frames ZIP file: {local_zip_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not cleanup frames ZIP file: {e}")

    async def handle_point_cloud_building(self, ch, method, model_id, data):
       
        try: 
            """Fase 2: Creazione del punto nuvola tramite Gaussian Splatting"""
            model_service.start_phase(model_id, "point_cloud_building")
            model = model_service.get_model_by_id(model_id)
            if not model:
                self.fail(model_id,"point_cloud_building",f"Error: No model found for model_id {model_id}")
                return False
            
            # 🆕 Salva backup delle immagini segmentate PRIMA di COLMAP
            model_dir = os.path.join(WORKING_DIR, f"{model_id}")
            input_dir = os.path.join(model_dir, 'input')

            if os.path.exists(input_dir) and os.listdir(input_dir):
                print(f"✅ Directory input già esistente per model_id {model_id}")
            else:
                print(f"📥 Scaricando directory input da S3 per model_id {model_id}")
                # Crea directory se non esiste
                os.makedirs(model_dir, exist_ok=True)

                # Determina ID iniziale da usare (retry o meno)
                s3_model_id = model_id if data.get('is_retry') is not None else model.parent_model_id

                # Prova a scaricare il file
                point_cloud_zip_s3_key = f"{S3_STAGING_PREFIX}/{s3_model_id}/{POINT_CLOUD_BUILDING_PHASE_ZIP_NAME}"
                success = phase_zip_helper.download_and_extract_phase_zip(point_cloud_zip_s3_key, model_dir)

                # Fallback: se fallisce, prova con parent_model_id (solo se diverso)
                if not success and s3_model_id != model.parent_model_id:
                    print("⚠️ Prima fase fallita, provo fallback con parent_model_id...")
                    fallback_s3_model_id = model.parent_model_id
                    fallback_point_cloud_zip_s3_key = f"{S3_STAGING_PREFIX}/{fallback_s3_model_id}/{POINT_CLOUD_BUILDING_PHASE_ZIP_NAME}"
                    success = phase_zip_helper.download_and_extract_phase_zip(fallback_point_cloud_zip_s3_key, model_dir)

                if not success:
                    self.fail(model_id, "point_cloud_building", 
                            f"Failed to download/extract point cloud building ZIP from {point_cloud_zip_s3_key}")
                    return False

                
                # Verifica nuovamente che l'estrazione sia andata a buon fine
                if not os.path.exists(input_dir):
                    self.fail(model_id, "point_cloud_building", 
                            f"Input directory is still invalid after ZIP extraction")
                    return False
            
            image_files = JobUtils.list_image_files(input_dir,{".jpg", ".jpeg", ".png"})
            frames_in = len(image_files)

            if frames_in == 0:
                self.fail(model_id, "point_cloud_building", "No image frames found in input directory")
                return False
            # Cartella sparse per la point cloud
            sparse_dir = os.path.join(model_dir, 'sparse')


            print(f"Generando la nuvola di punti per model_id {model_id}...")
                
            print(f"🔄 Avvio generazione point cloud per model_id {model_id}...")
            colmap_start_time = datetime.utcnow()
            
            # COLMAP API call
            convert_request = {"input_dir": model_dir}
            response = requests.post(COLMAP_API_URL + "/convert", json=convert_request)

            # ⏱️ FINE TIMING COLMAP
            colmap_end_time = datetime.utcnow()
            colmap_duration = colmap_end_time - colmap_start_time
            colmap_duration_seconds = colmap_duration.total_seconds()

            if response.status_code != 200:
                self.fail(model_id,"point_cloud_building",f" Colmap error: {response.text}")
                return False
            
            # 5. Verifica che COLMAP abbia creato i file
            if not os.path.exists(sparse_dir) or not os.listdir(sparse_dir):
                self.fail(model_id,"point_cloud_building",f"Error: no cloud point created for model_id {model_id}")
                return False   
            
            print(f"✅ Nuvola di punti generata con successo per model_id {model_id}")

           

            is_zip_uploaded = phase_zip_helper.create_phase_zip_and_upload(model_id,model_dir,TRAINING_PHASE_ZIP_NAME,['sparse', 'images'])
            if not is_zip_uploaded:
                self.fail(model_id,"training",f"Error: Failed to create training phase ZIP for model_id {model_id}")
                return False

            reconstruction_params = point_cloud_utils.aggregate_reconstruction_stats(model_dir)

            frame_per_second = colmap_duration_seconds/frames_in
            thousand_points_per_second = colmap_duration_seconds/(reconstruction_params['generated_points']/1000)

            # 7. Aggiorna stato e passa alla fase successiva
            phase_metadata = {
                "reconstruction_params": reconstruction_params,
                "colmap_duration_seconds": round(colmap_duration_seconds, 2),
                "colmap_start_time": colmap_start_time.isoformat(),
                "colmap_end_time": colmap_end_time.isoformat(),
                "frame_per_second":frame_per_second,
                "thousand_points_per_second":thousand_points_per_second
            }

            model_service.complete_phase(model_id, "point_cloud_building", "PENDING",
                                    metadata=phase_metadata)
        
            return True
        except Exception as e:
            self.fail(model_id,"training",f"Training error: {e}")
            return False

    async def handle_training(self, ch, method, model_id, data):
        print(f"🚀 Start training model {model_id}")
        
        # ⏰ START FASE
        model_service.start_phase(model_id, "training")
        
        try:
            model = model_service.get_model_by_id(model_id)
            if not model:
                self.fail(model_id,"training",f"Error: No model found for model_id {model_id}")
                return False
            
            model_dir = os.path.join(WORKING_DIR, f"{model_id}")
            image_dir = os.path.join(model_dir, "images")
            sparse_dir = os.path.join(model_dir, "sparse")
            
            if os.path.exists(image_dir) and os.listdir(image_dir) and os.path.exists(sparse_dir) and os.listdir(sparse_dir):
                print(f"✅ Directories images e sparse già esistenti per model_id {model_id}")
            else:
                print(f"📥 Scaricando directory input da S3 per model_id {model_id}")
                # Crea directory se non esiste
                os.makedirs(model_dir, exist_ok=True)

                 # Determina ID iniziale da usare (retry o meno)
                s3_model_id = model_id if data.get('is_retry') is not None else model.parent_model_id

                # Prova a scaricare il file
                training_zip_s3_key = f"{S3_STAGING_PREFIX}/{s3_model_id}/{TRAINING_PHASE_ZIP_NAME}"
                success = phase_zip_helper.download_and_extract_phase_zip(training_zip_s3_key, model_dir)

                # Fallback: se fallisce, prova con parent_model_id (solo se diverso)
                if not success and s3_model_id != model.parent_model_id:
                    print("⚠️ Prima fase fallita, provo fallback con parent_model_id...")
                    fallback_s3_model_id = model.parent_model_id
                    fallback_training_zip_s3_key = f"{S3_STAGING_PREFIX}/{fallback_s3_model_id}/{TRAINING_PHASE_ZIP_NAME}"
                    success = phase_zip_helper.download_and_extract_phase_zip(fallback_training_zip_s3_key, model_dir)

                # Scarica ed estrai lo ZIP
                if not success:
                    self.fail(model_id, "training", 
                            f"Failed to download/extract training ZIP from {training_zip_s3_key}")
                    return False

                # Verifica nuovamente che l'estrazione sia andata a buon fine
                if not os.path.exists(image_dir)  or not os.path.exists(sparse_dir):
                    self.fail(model_id, "training", 
                            f"Images or sparse directories are still invalid after ZIP extraction")
                    return False
 

            engine = model.training_config.get('engine') if model.training_config else None
            quality_level = model.training_config.get('quality_level') if model.training_config else None
            
            # 🆕 INIZIALIZZA VRAM MONITOR A LIVELLO HOST
            from utils.vram_monitor import VRAMMonitor
            vram_monitor = VRAMMonitor(interval=2.0,log_directory=model_dir)

             # 🚀 AVVIA MONITORING PRIMA DELLA CHIAMATA API
            vram_monitor.start_monitoring()

            print(f"🎯 Starting training with engine: {engine}")
            api_url = engine_map.get(engine, {}).get('api-url')
            if not api_url:
                self.fail(model_id,"training",f"Error: No api url found for engine {engine}")
                return False

            depth_reg_start_time = None
            depth_reg_end_time = None
            depth_reg_duration_seconds = None
            if engine == 'INRIA':
                depth_reg_start_time = datetime.utcnow()
                depth_request = { "input_dir": model_dir}
                response = requests.post(f"{api_url}/depth_regularization", json=depth_request)
                depth_reg_end_time = datetime.utcnow()
                depth_reg_duration = depth_reg_end_time - depth_reg_start_time
                depth_reg_duration_seconds = depth_reg_duration.total_seconds()
                response.raise_for_status()  # Controlla che non ci siano errori nel render
            

            if not engine:
                self.fail(model_id,"training",f"Error: No engine found in model {model_id}")
                return False
            
            # 3. Prepara directory di output
            train_output_folder = os.path.join(model_dir, 'output')
            os.makedirs(train_output_folder, exist_ok=True)
            
            generated_params = training_params_service.generate_params(Engine(engine),QualityLevel(quality_level))

            training_start_time = datetime.utcnow()
            # 🆕 STIMA GAUSSIANE BASATA SUI PARAMETRI DI RICOSTRUZIONE
            if engine == 'MCMC' or engine == 'TAMING':
                image_dir = os.path.join(model_dir, "images")
                try:
                    # Estrai i parametri di ricostruzione dalla fase point_cloud_building
                    reconstruction_params = model.phases.get('point_cloud_building', {}).metadata['reconstruction_params']
                    
                    if reconstruction_params:
                        # Usa la formula semplice (più affidabile)
                        res1 = estimate_gaussians_from_stats(reconstruction_params,
                                                             image_dir,
                                                             True,False,False,False,False, )
                        res2 = estimate_gaussians_from_stats(reconstruction_params,image_dir,True,True,False,False,False )
                        res3 = estimate_gaussians_from_stats(reconstruction_params,image_dir,True,True,True,False,False )
                        res4 = estimate_gaussians_from_stats(reconstruction_params,image_dir,True,True,True,True, False  )
                        res5 = estimate_gaussians_from_stats(reconstruction_params, image_dir)

                        logger.info("Estimate debug:\n%s", json.dumps(res1, indent=2))
                        logger.info("Estimate debug:\n%s", json.dumps(res2, indent=2))
                        logger.info("Estimate debug:\n%s", json.dumps(res3, indent=2))
                        logger.info("Estimate debug:\n%s", json.dumps(res4, indent=2))
                        logger.info("Estimate debug:\n%s", json.dumps(res5, indent=2))
                        suggested_params = res5['suggested_params']
                        #estimated_gaussians = JobUtils.estimate_final_gaussians(reconstruction_params['generated_points'])
                        
                        logger.info("Suggested params:\n%s", json.dumps(suggested_params, indent=2))

                        
                        # Applica la stima in base all'algoritmo
                        if engine == 'MCMC':
                            # Per MCMC, usa la stima come cap_max (con un margine di sicurezza)
                            #original_cap_max = generated_params.final_params.get('cap_max', suggested_params['cap_max'])
                            # Usa il minimo tra la stima (+20% margine) e il valore configurato
                            generated_params.final_params['cap_max'] = suggested_params['cap_max']
                            print(f"🎯 MCMC cap_max set to: {generated_params.final_params['cap_max']:,}")
                            
                        elif engine == 'TAMING':
                            generated_params.final_params['budget'] = suggested_params['budget']
                            # Per TAMING, usa la stima come budget (con un margine di sicurezza)
                            # Usa il minimo tra la stima (+20% margine) e il valore configurato
                            print(f"🎯 TAMING budget set to: {generated_params.final_params['budget']:,}")
                    else:
                        print(f"⚠️ No reconstruction params found, using default values")
                        
                except Exception as e:
                    print(f"⚠️ Error estimating gaussians: {e}, using default values")
                    # Continua con i valori di default se c'è un errore
            
            # 4. Prepara richiesta di training
            train_request = {
                "input_dir": model_dir,
                "output_dir": train_output_folder,
                "params": generated_params.final_params,
                "has_depths": True
            }
        
            # TRAIN API call
            response = requests.post(f"{api_url}/train", json=train_request)
            # ⏱️ FINE TIMING TRAIN
            training_end_time = datetime.utcnow()
            training_duration = training_end_time - training_start_time
            training_duration_seconds = training_duration.total_seconds()
            
            if response.status_code != 200:
                vram_monitor.stop_monitoring()
                self.fail(model_id,"training",f"Error: Training failed status code {response.status_code}")
                return False
            
            print(f"✅ Training completato con successo")
            
            # 📊 FERMA MONITORING E RACCOGLI STATS
            vram_stats = vram_monitor.stop_monitoring()

            
            # Prima dell'upload
            print(f"🔍 Output folder exists: {os.path.exists(train_output_folder)}")

            iterations = generated_params.final_params["iterations"]
            image_files = JobUtils.list_image_files(image_dir,{".jpg", ".jpeg", ".png"})
            # 1. Verifica se sparse directory esiste (output di COLMAP)
            output_dir = os.path.join(model_dir, 'output')
            ply_path = job_utils.find_latest_iteration_folder(output_dir)
             # Leggi il file PLY
            plydata = PlyData.read(ply_path)
            vert = len(plydata["vertex"])
        
            frames_in = len(image_files)
            time_per_1k_iter_s = training_duration_seconds/(iterations/1000)
            time_per_1k_frames_s = training_duration_seconds/frames_in
            time_per_100k_gauss_s = training_duration_seconds/(vert/100000)
            depth_share = depth_reg_duration_seconds/frames_in if depth_reg_duration_seconds is not None else None
            # ✅ COMPLETE FASE
            phase_metadata = {
            "final_params": generated_params.final_params,
            "depth_reg_duration_seconds": round(depth_reg_duration_seconds, 2) if depth_reg_duration_seconds is not None else None,
            "training_duration_seconds": round(training_duration_seconds, 2),
            "training_start_time": training_start_time.isoformat(),
            "training_end_time": training_end_time.isoformat(),
            "depth_reg_start_time": depth_reg_start_time.isoformat() if depth_reg_start_time is not None else None,
            "depth_reg_end_time": depth_reg_end_time.isoformat() if depth_reg_end_time is not None else None,
            "time_per_1k_iter_s":time_per_1k_iter_s,
            "time_per_1k_frames_s":time_per_1k_frames_s,
            "time_per_100k_gauss_s":time_per_100k_gauss_s,
            "depth_share":depth_share,
            "vram_stats": vram_stats  # ← DIRETTAMENTE NEI METADATI MONGODB
            }

            model_service.complete_phase(model_id, "training", "PENDING",
                                metadata=phase_metadata)
            return True
        
        except Exception as e:
            # 🛑 FERMA MONITORING ANCHE IN CASO DI ERRORE
            vram_monitor.stop_monitoring()
        
            self.fail(model_id,"training",f"Training error: {e}")
        return False

    async def handle_model_upload(self, ch, method, model_id, data):
        """Fase 4: Caricamento del modello addestrato su S3"""
        model_service.start_phase(model_id, "upload")

        try:
            model = model_service.get_model_by_id(model_id)
            if not model:
                self.fail(model_id, "training", f"Error: No model found for model_id {model_id}")
                return False

            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = os.path.join(WORKING_DIR, f"{model_id}")

                output_dir = os.path.join(model_dir, "output")
                if not os.path.isdir(output_dir):
                    self.fail(model_id, "upload", f"No 'output' folder found at: {output_dir}")
                    return False

                # Trova il PLY dell’ultima iterazione
                ply_path = job_utils.find_latest_iteration_folder(output_dir)
                if not ply_path or not os.path.isfile(ply_path):
                    self.fail(model_id, "upload", f"No PLY file found under: {output_dir}")
                    return False

                cameras_file_path = os.path.join(output_dir, "cameras.json")
                if not os.path.isfile(cameras_file_path):
                    self.fail(model_id, "upload", f"'cameras.json' not found at: {cameras_file_path}")
                    return False

                engine = model.training_config.get('engine') if model.training_config else None
                is_taming = (engine == 'TAMING')

                # Dove è installato @mkkellogg/gaussian-splats-3d (la cartella con node_modules)
                node_cwd = os.getenv("NODE_WORKDIR", WORKING_DIR)  # oppure passalo esplicitamente

                gsplat_path = os.path.join(temp_dir, "point_cloud.ksplat")

                # PLY -> KSPLAT
                ply_to_splat_start_time = datetime.utcnow()
                try:
                    data_bytes = process_ply_to_ksplat(
                        ply_path,
                        convert_taming_opacity=is_taming,
                        sh_degree=2,
                        compression_level=1
                    )
                except Exception as e:
                    self.fail(model_id, "upload", f"PLY→KSPLAT conversion failed: {e}")
                    return False

                save_splat_file(data_bytes, gsplat_path)
                ply_to_splat_end_time = datetime.utcnow()
                ply_to_splat_duration_seconds = (ply_to_splat_end_time - ply_to_splat_start_time).total_seconds()

                # Aggiungi cameras.json alla temp dir
                shutil.copy2(cameras_file_path, temp_dir)

                # Crea ZIP da tutto il temp_dir
                zip_path_noext = os.path.join(model_dir, "3d_model")
                zip_filename = shutil.make_archive(zip_path_noext, "zip", temp_dir)  # ritorna il path completo

                # Upload S3
                print(f"Inizio del caricamento del modello {model_id} su S3...")
                zip_model_suffix = "3d_model.zip"
                zip_model_s3_key = f"{S3_DELIVERY_PREFIX}/{model_id}/{zip_model_suffix}"
                try:
                    repository_service.upload(zip_filename, zip_model_s3_key)
                    print(f"✅ Upload ok: s3://{zip_model_s3_key}")
                except Exception as e:
                    self.fail(model_id, "upload", f"Errore durante l'upload su S3: {e}")
                    return False

                # Metriche
                ply_size_mb = round(os.path.getsize(ply_path) / (1024 * 1024), 3)
                splat_size_mb = round(os.path.getsize(gsplat_path) / (1024 * 1024), 3)
                time_per_mb = round(ply_to_splat_duration_seconds / max(ply_size_mb, 1e-9), 6)
                compression_ratio = round(splat_size_mb / max(ply_size_mb, 1e-9), 3)

                phase_metadata = {
                    "ply_to_splat_duration_seconds": round(ply_to_splat_duration_seconds, 2),
                    "ply_to_splat_start_time": ply_to_splat_start_time.isoformat(),
                    "ply_to_splat_end_time": ply_to_splat_end_time.isoformat(),
                    "ply_size_mb": ply_size_mb,
                    "splat_size_mb": splat_size_mb,
                    "time_per_mb": time_per_mb,
                    "compression_ratio": compression_ratio
                }

                model_service.complete_phase(model_id, "upload", "PENDING", metadata=phase_metadata)
                model_service.update_model_status(model_id, {"zip_model_suffix": zip_model_suffix})
                print(f"Model {model_id} caricato su S3 con successo!")
                return True

        except FileNotFoundError as e:
            self.fail(model_id, "upload", f"Error: {e}")
            return False
        except NoCredentialsError:
            self.fail(model_id, "upload", "Error: AWS credential not found.")
            return False
        except PartialCredentialsError:
            self.fail(model_id, "upload", "Error: AWS credential partially not found")
            return False
        except Exception as e:
            self.fail(model_id, "upload", f"Fail to upload model: {e}")
            return False
        
    
        
    async def handle_metrics_generation(self, ch, method, model_id, data):
        """Fase 5: Generazione delle metriche e salvataggio su Mongo"""
         # ⏰ START FASE
        model_service.start_phase(model_id, "metrics_evaluation")
        
        try:
            model = model_service.get_model_by_id(model_id)
            if not model:
                self.fail(model_id,"metrics_evaluation",f"Error: No model found for model_id {model_id}")
                return False
            
            model_dir = os.path.join(WORKING_DIR, f"{model_id}")
                
            # 1. Verifica se sparse directory esiste (output del training)
            output_dir = os.path.join(model_dir, 'output')
            if not os.path.exists(output_dir):
                self.fail(model_id,"metrics_evaluation",f"No folder output found: {e}")
                return False

            engine = model.training_config.get('engine') if model.training_config else 'INRIA'

            render_request = { "output_dir": output_dir}
            response = requests.post(engine_map.get(engine).get('api-url') + "/render", json=render_request)
            response.raise_for_status()  # Controlla che non ci siano errori nel render

            metrics_request = { "output_dir": output_dir}
            response = requests.post(engine_map.get(engine).get('api-url') + "/metrics", json=metrics_request)
            response.raise_for_status()  # Controlla che non ci siano errori nel render

            # Verifica che il file results.json esista
            results_json_path = os.path.join(output_dir, "results.json")
            if not os.path.exists(results_json_path):
                raise FileNotFoundError("Il file 'results.json' non è stato trovato.")
            # Leggi il contenuto di results.json
            with open(results_json_path, 'r') as f:
                results_data = json.load(f)

            # Strategia 2: Cerca chiave che inizia con "ours_"
            ours_keys = [key for key in results_data.keys() if key.startswith("ours_")]
            results = None
            if len(ours_keys) == 1:
                # Solo una chiave "ours_", usala
                key = ours_keys[0]
                print(f"✅ Found single ours key: {key}")
                results = job_utils._extract_metrics_from_section(results_data[key])
            elif len(ours_keys) > 1:
                def extract_number(key):
                    try:
                        return int(key.split("_")[1])
                    except (IndexError, ValueError):
                        return 0
                latest_key = max(ours_keys, key=extract_number)
                print(f"✅ Found multiple ours keys, using latest: {latest_key}")
                results = job_utils._extract_metrics_from_section(results_data[latest_key])
            else:
                # Nessuna chiave "ours_", cerca qualsiasi chiave con metriche
                for key, value in results_data.items():
                    if isinstance(value, dict) and any(metric in value for metric in ["SSIM", "PSNR", "LPIPS"]):
                        print(f"✅ Found metrics in fallback key: {key}")
                        results = job_utils._extract_metrics_from_section(value)
                
                raise KeyError(f"Nessuna sezione con metriche trovata nel file results.json. Chiavi disponibili: {list(results_data.keys())}")
            
            model_service.complete_phase(model_id, "metrics_evaluation","COMPLETED",metadata={"metrics": results})
            print(f"Model {model_id} caricato su S3 con successo!")
            return True  # Successo
        except FileNotFoundError as e:
            self.fail(model_id,"metrics_evaluation",f"Error: {e}")
            return False  # Fallimento
        except NoCredentialsError:
            self.fail(model_id,"metrics_evaluation","Error: AWS credential not found.")
            return False  # Fallimento
        except PartialCredentialsError:
            self.fail(model_id,"metrics_evaluation","Error: AWS credential partially not found")
            return False  # Fallimento
        except Exception as e:
            self.fail(model_id,"metrics_evaluation",f"Fail to load model: {e}")
            return False  # Fallimento
        
    
