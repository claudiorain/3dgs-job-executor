from config.message_queue import get_connection,get_channel  # Assicurati che questa funzione restituisca il client del database
from services.model_service import ModelService
from services.repository_service import RepositoryService
from services.training_params_service import TrainingParamsService,QualityLevel
from utils.video_frame_utils import FrameExtractor
from models.model import Engine
from converters.ply_to_gsplat_converter import save_splat_file,process_ply_to_splat
from datetime import datetime
import subprocess
import os
import json
import requests
import shutil
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import tempfile
from utils.phase_zip_helper import PhaseZipHelper
from utils.job_utils import JobUtils

model_service = ModelService()
repository_service = RepositoryService()
frame_extractor = FrameExtractor()
training_params_service = TrainingParamsService()
phase_zip_helper = PhaseZipHelper()
job_utils = JobUtils()


quality_frame_targets = {
    QualityLevel.FAST: 150,
    QualityLevel.BALANCED: 200,
    QualityLevel.QUALITY: 250,
}

# Cartella per i file di staging (zip delle fasi)
S3_STAGING_PREFIX = os.getenv('S3_STAGING_PREFIX', 'staging')

# Cartella per i deliverable finali
S3_DELIVERY_PREFIX = os.getenv('S3_DELIVERY_PREFIX', 'delivery')

WORKING_DIR = os.getenv("MODEL_WORKING_DIR") 
GAUSSIAN_SPLATTING_API_URL = "http://gaussian-splatting-api:8050"
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
        Fase 1: Download del video e creazione dei fotogrammi con preprocessing atomico
        REFACTORED: Utilizza VideoFrameExtractionService per operazioni atomiche
        """
        from services.video_frame_extraction_service import (
            VideoFrameExtractionService, 
            FrameExtractionParams
        )
        
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

        # Percorso fisso per sfruttare la cache
        local_video_path = os.path.join(model_dir, 'input_video.mp4')
        frames_output_folder = os.path.join(model_dir, 'input')
        
        try:
            # ===== STEP 1: DOWNLOAD VIDEO =====
            print(f"📥 Downloading video from S3: {video_s3_key}")
            repository_service.download(video_s3_key, local_video_path)
                        
            # Verifica download
            if not os.path.exists(local_video_path):
                raise Exception(f"Video file not found after download: {local_video_path}")
            
            video_size = os.path.getsize(local_video_path)
            print(f"📊 Video downloaded successfully: {video_size / 1024 / 1024:.2f} MB")
            
            # ===== STEP 2: RECUPERO PARAMETRI DI CONFIGURAZIONE =====
            engine = model.training_config.get('engine') if model.training_config else None
            quality_level = model.training_config.get('quality_level') if model.training_config else None
            
            if not engine:
                self.fail(model_id, "frame_extraction", f"Error: No engine found in model {model_id}")
                return False  

            # Genera parametri ottimizzati per engine e hardware
            generated_params = training_params_service.generate_params(Engine(engine), QualityLevel(quality_level))
            print(f"🎯 Generated training parameters completed")
            
            # Estrai preprocessing_params per frame extraction
            target_width = generated_params.preprocessing_params.get('target_width', 1280)
            target_height = generated_params.preprocessing_params.get('target_height', 720)
            target_frame_count = generated_params.preprocessing_params.get('target_frames', 200)
            
            print(f"🔍 Frame extraction config - Width: {target_width}, Height: {target_height}, Frames: {target_frame_count}")
            
            # ===== STEP 3: ESTRAZIONE ATOMICA DEI FRAME =====
            print(f"🎬 Starting atomic frame extraction...")
            
            # Inizializza il service di estrazione
            frame_extraction_service = VideoFrameExtractionService()
            
            # Configura parametri di estrazione
            extraction_params = FrameExtractionParams(
                target_width=target_width,
                target_height=target_height,
                target_frame_count=target_frame_count,
                selection_method="best-n",  # Ottimale per COLMAP
                min_buffer="1"  # Gap minimale tra frame consecutivi
            )
            
            # Esecuzione atomica dell'estrazione
            extraction_result = frame_extraction_service.extract_frames(
                video_path=local_video_path,
                output_directory=frames_output_folder,
                extraction_params=extraction_params
            )
            
            # Verifica successo operazione atomica
            if not extraction_result.success:
                raise Exception(f"Frame extraction failed: {extraction_result.error_message}")
            
            print(f"✅ Atomic frame extraction completed: {extraction_result.extracted_frame_count} frames")
            
            # ===== STEP 4: GESTIONE THUMBNAIL =====
            thumbnail_suffix = "thumbnail.jpg"
            thumbnail_s3_key = f"{S3_DELIVERY_PREFIX}/{model_id}/{thumbnail_suffix}"
            
            # Usa il primo frame come thumbnail
            if extraction_result.frame_files:
                thumbnail_path = extraction_result.frame_files[0]
                print(f"📸 Using first frame as thumbnail: {thumbnail_path}")
                
                if os.path.exists(thumbnail_path):
                    try:
                        repository_service.upload(thumbnail_path, thumbnail_s3_key)
                        print(f"✅ Thumbnail uploaded to S3: {thumbnail_s3_key}")
                    except Exception as e:
                        print(f"❌ Error uploading thumbnail: {e}")
                        thumbnail_suffix = None
                else:
                    print(f"⚠️ Thumbnail file not found: {thumbnail_path}")
                    thumbnail_suffix = None
            else:
                print(f"⚠️ No frames available for thumbnail")
                thumbnail_suffix = None

            # ===== STEP 5: UPLOAD FASE INTERMEDIA SU S3 =====
            print(f"📤 Uploading frame extraction results to S3...")
            is_zip_uploaded = phase_zip_helper.create_phase_zip_and_upload(
                model_id, model_dir, POINT_CLOUD_BUILDING_PHASE_ZIP_NAME, ['input']
            )
            
            if not is_zip_uploaded:
                self.fail(model_id, "frame_extraction", f"Error: Failed to create phase ZIP for model_id {model_id}")
                return False

            # ===== STEP 6: COMPLETAMENTO FASE CON METADATI =====
            phase_metadata = {
                "frame_count": extraction_result.extracted_frame_count,
                "video_size_mb": video_size / 1024 / 1024,
                "extraction_parameters": extraction_result.extraction_params,
                "processing_stats": extraction_result.processing_stats,
                "extraction_service_version": "atomic_v1.0"
            }

            model_service.complete_phase(model_id, "frame_extraction", metadata=phase_metadata)
            
            # Aggiorna thumbnail nel modello se disponibile
            if thumbnail_suffix:
                model_service.update_model_status(model_id, {
                    "thumbnail_suffix": thumbnail_suffix
                })
            
            print(f"✅ Frame extraction phase completed successfully")
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

                # Ottieni il percorso S3 del ZIP della fase point cloud building
                point_cloud_zip_s3_key = f"{S3_STAGING_PREFIX}/{model.parent_model_id}/{POINT_CLOUD_BUILDING_PHASE_ZIP_NAME}"

                # Scarica ed estrai lo ZIP
                success = phase_zip_helper.download_and_extract_phase_zip(point_cloud_zip_s3_key, model_dir)

                if not success:
                    self.fail(model_id, "point_cloud_building", 
                            f"Failed to download/extract point cloud building ZIP from {point_cloud_zip_s3_key}")
                    return False
                
                # Verifica nuovamente che l'estrazione sia andata a buon fine
                if not os.path.exists(input_dir):
                    self.fail(model_id, "point_cloud_building", 
                            f"Input directory is still invalid after ZIP extraction")
                    return False
            
            # Cartella sparse per la point cloud
            sparse_dir = os.path.join(model_dir, 'sparse')


            print(f"Generando la nuvola di punti per model_id {model_id}...")
                
            print(f"🔄 Avvio generazione point cloud per model_id {model_id}...")
            colmap_start_time = datetime.utcnow()
            
            # COLMAP API call
            convert_request = {"input_dir": model_dir}
            response = requests.post(COLMAP_API_URL + "/convert", json=convert_request)

            response_json = response.json()
            reconstruction_params = response_json.get("reconstruction_params", {})
            print(f"🔍 Extracted reconstruction_params: {reconstruction_params}")
    
            # 🆕 RACCOGLI STATISTICHE COLMAP
            reconstruction_params["generated_points"] = job_utils.get_colmap_reconstruction_stats(model_dir)
    
            print(f"🔍 Final reconstruction_params after adding generated_points: {reconstruction_params}")

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

           
            print(f"📊 COLMAP Stats: {reconstruction_params}")

            is_zip_uploaded = phase_zip_helper.create_phase_zip_and_upload(model_id,model_dir,TRAINING_PHASE_ZIP_NAME,['sparse', 'images'])
            if not is_zip_uploaded:
                self.fail(model_id,"training",f"Error: Failed to create training phase ZIP for model_id {model_id}")
                return False

            # 7. Aggiorna stato e passa alla fase successiva
            phase_metadata = {
                "reconstruction_params": reconstruction_params,
                "colmap_duration_seconds": round(colmap_duration_seconds, 2),
                "colmap_start_time": colmap_start_time.isoformat(),
                "colmap_end_time": colmap_end_time.isoformat(),
            }

            model_service.complete_phase(model_id, "point_cloud_building", 
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

                # Ottieni il percorso S3 del ZIP della fase point cloud building
                training_zip_s3_key = f"{S3_STAGING_PREFIX}/{model.parent_model_id}/{TRAINING_PHASE_ZIP_NAME}"

                # Scarica ed estrai lo ZIP
                success = phase_zip_helper.download_and_extract_phase_zip(training_zip_s3_key, model_dir)
                if not success:
                    self.fail(model_id, "training", 
                            f"Failed to download/extract training ZIP from {training_zip_s3_key}")
                    return False

            
             # Verifica nuovamente che l'estrazione sia andata a buon fine
            if not os.path.exists(image_dir)  or not os.path.exists(sparse_dir):
                self.fail(model_id, "training", 
                         f"Images or sparse directories are still invalid after ZIP extraction")
                return False
            
            
           
            training_start_time = datetime.utcnow()

            engine = model.training_config.get('engine') if model.training_config else None
            quality_level = model.training_config.get('quality_level') if model.training_config else None
            
            print(f"🎯 Starting training with engine: {engine}")
            api_url = engine_map.get(engine, {}).get('api-url')
            if not api_url:
                self.fail(model_id,"training",f"Error: No api url found for engine {engine}")
                return False

            if engine == 'INRIA':
                depth_request = { "input_dir": model_dir}
                response = requests.post(f"{api_url}/depth_regularization", json=depth_request)
                response.raise_for_status()  # Controlla che non ci siano errori nel render
            

            if not engine:
                self.fail(model_id,"training",f"Error: No engine found in model {model_id}")
                return False
            
            # 3. Prepara directory di output
            train_output_folder = os.path.join(model_dir, 'output')
            os.makedirs(train_output_folder, exist_ok=True)
            
            generated_params = training_params_service.generate_params(Engine(engine),QualityLevel(quality_level))
           
            # 🆕 STIMA GAUSSIANE BASATA SUI PARAMETRI DI RICOSTRUZIONE
            if engine == 'MCMC' or engine == 'TAMING':
                try:
                    # Estrai i parametri di ricostruzione dalla fase point_cloud_building
                    reconstruction_params = model.phases.get('point_cloud_building', {}).metadata['reconstruction_params']
                    
                    if reconstruction_params and 'generated_points' in reconstruction_params:
                        # Usa la formula semplice (più affidabile)
                        estimated_gaussians = JobUtils.estimate_final_gaussians(reconstruction_params['generated_points'])
                        
                        print(f"📊 Estimated gaussians: {estimated_gaussians:,} based on {reconstruction_params['generated_points']:,} points")
                        
                        # Applica la stima in base all'algoritmo
                        if engine == 'MCMC':
                            # Per MCMC, usa la stima come cap_max (con un margine di sicurezza)
                            original_cap_max = generated_params.final_params.get('cap_max', 800000)
                            # Usa il minimo tra la stima (+20% margine) e il valore configurato
                            generated_params.final_params['cap_max'] = min(
                                int(estimated_gaussians * 1.2), 
                                original_cap_max
                            )
                            print(f"🎯 MCMC cap_max set to: {generated_params.final_params['cap_max']:,}")
                            
                        elif engine == 'TAMING':
                            # Per TAMING, usa la stima come budget (con un margine di sicurezza)
                            original_budget = generated_params.final_params.get('budget', 1000000)
                            # Usa il minimo tra la stima (+20% margine) e il valore configurato
                            generated_params.final_params['budget'] = min(
                                int(estimated_gaussians * 1.2), 
                                original_budget
                            )
                            print(f"🎯 TAMING budget set to: {generated_params.final_params['budget']:,}")
                    else:
                        print(f"⚠️ No reconstruction params found, using default values")
                        
                except Exception as e:
                    print(f"⚠️ Error estimating gaussians: {e}, using default values")
                    # Continua con i valori di default se c'è un errore
            

           # 📌 Aggiorna fase con i parametri di training PRIMA dell'esecuzione vera e propria
            model_service.update_phase(
                model_id,
                "training",
                metadata={
                    "training_parameters": {
                        "engine": engine,
                        "quality_level": quality_level,
                        "final_params": generated_params.final_params
                    }
                }
            )
            # 4. Prepara richiesta di training
            train_request = {
                "input_dir": model_dir,
                "output_dir": train_output_folder,
                "params": generated_params.final_params,
            }
        
            # TRAIN API call
            response = requests.post(f"{api_url}/train", json=train_request)
            # ⏱️ FINE TIMING TRAIN
            training_end_time = datetime.utcnow()
            training_duration = training_end_time - training_start_time
            training_duration_seconds = training_duration.total_seconds()
            
            if response.status_code != 200:
                self.fail(model_id,"training",f"Error: Training failed status code {response.status_code}")
                return False
            
            print(f"✅ Training completato con successo")
            
            # Prima dell'upload
            print(f"🔍 Output folder exists: {os.path.exists(train_output_folder)}")

            # ✅ COMPLETE FASE
            phase_metadata = {
            "training_duration_seconds": round(training_duration_seconds, 2),
            "training_start_time": training_start_time.isoformat(),
            "training_end_time": training_end_time.isoformat()
            }

            model_service.complete_phase(model_id, "training", 
                                   metadata=phase_metadata)
            return True
        
        except Exception as e:
            self.fail(model_id,"training",f"Training error: {e}")
            return False

    async def handle_model_upload(self, ch, method, model_id, data):
        """Fase 4: Caricamento del modello addestrato su S3"""

        # ⏰ START FASE
        model_service.start_phase(model_id, "upload")
        
        try:
            model = model_service.get_model_by_id(model_id)
            if not model:
                self.fail(model_id,"training",f"Error: No model found for model_id {model_id}")
                return False
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Crea i percorsi come al solito, ma per gsplat_path utilizziamo la cartella temporanea
                model_dir = os.path.join(WORKING_DIR, f"{model_id}")
                
                # 1. Verifica se sparse directory esiste (output di COLMAP)
                output_dir = os.path.join(model_dir, 'output')
                if not os.path.exists(output_dir):
                    self.fail(model_id,"upload",f"No folder output found: {e}")
                    return False


                ply_path = job_utils.find_latest_iteration_folder(output_dir)
                cameras_file_path = os.path.join(output_dir, "cameras.json")
            
                engine = model.training_config.get('engine') if model.training_config else None

                # Usa la cartella temporanea per il file .splat
                gsplat_path = os.path.join(temp_dir, "point_cloud.splat")
                # Processa il file .ply e salva il .splat nella cartella temporanea
                save_splat_file(process_ply_to_splat(ply_path,engine == Engine.TAMING.value), gsplat_path)

                # Aggiungi il file cameras.json alla cartella temporanea
                shutil.copy(cameras_file_path, temp_dir)

                # Creazione del file ZIP che contiene il file .splat e il file cameras.json
                zip_filename = os.path.join(model_dir, "3d_model.zip")
                shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', temp_dir)

                # Inizio del caricamento del modello su S3
                print(f"Inizio del caricamento del modello {model_id} su S3...")
                zip_model_suffix = f"3d_model.zip"
                zip_model_s3_key = f"{S3_DELIVERY_PREFIX}/{model_id}/{zip_model_suffix}"

                try:
                    repository_service.upload(zip_filename, zip_model_s3_key)
                    print(f"✅ Il modello {model_id} è stato caricato con successo su S3 nella chiave {zip_model_s3_key}")
                except Exception as e:
                    self.fail(model_id,"upload",f"Errore durante l'upload su S3: {e}")
                    return False
                
               
                # Passa al prossimo job (upload)
                model_service.complete_phase(model_id, "upload")

                model_service.update_model_status(model_id, {"zip_model_suffix": zip_model_suffix})
                print(f"Model {model_id} caricato su S3 con successo!")
                return True
                
        except FileNotFoundError as e:
            self.fail(model_id,"upload",f"Error: {e}")
            return False
        except NoCredentialsError:
            self.fail(model_id,"upload","Error: AWS credential not found.")
            return False
        except PartialCredentialsError:
            self.fail(model_id,"upload","Error: AWS credential partially not found")
            return False
        except Exception as e:
            self.fail(model_id,"upload",f"Fail to upload model: {e}")
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
            
            model_service.complete_phase(model_id, "metrics_evaluation",overall_status="COMPLETED",metadata={"metrics": results})
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
        
    
